/*
 * attn_omp.c -- Q4: attn_best.c parallelised with OpenMP.
 *
 * The kernel (6 x 64 register tiles, online softmax, vectorised exp2) is the same
 * as attn_best.c. Query blocks are independent (each owns its rows of O and its
 * running max/sum), so they are distributed over threads with no synchronisation
 * except the implicit barriers after packing and at the end:
 *
 *   - K panels and the aligned copy of V are packed in parallel (omp for).
 *   - The block size adapts to the thread count: about 2 blocks per thread,
 *     a multiple of 6 rows, at most 96 rows. Blocks are handed out with
 *     schedule(dynamic, 1), so a thread on a slow or shared core takes fewer blocks.
 *   - Each thread has its own O accumulator, running max/sum and Q tail buffer.
 *   - K and V (0.5-2 MB) are read by every thread; they stay in each core's L2 /
 *     socket's L3, so the run remains compute-bound up to all cores.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <immintrin.h>

#define MR 6  /* query rows per register strip              */
#define BC 64 /* keys per panel = columns of the S tile       */
#define BR_MAX 96 /* largest query block (multiple of MR); the actual size adapts to the thread count */
#define NEG_BIG (-1000.0f) /* "minus infinity" that keeps exp2_ps NaN-free */

/* 2^x, 16 lanes, for finite x <= 0 (x < -149 returns 0). */
static inline __attribute__((always_inline)) __m512 exp2_ps(__m512 x)
{
    const __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(x, n); /* f in [-0.5, 0.5] */
    __m512 p = _mm512_set1_ps(1.327647129073739e-3f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.675541892647743e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.550713092088699e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022120237350464e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.931469440460205e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0000001192092896f));
    return _mm512_scalef_ps(p, n); /* p * 2^n */
}

/* One 6-row strip against one 64-key panel: scores, online softmax, P*V. */
static inline __attribute__((always_inline)) void strip_panel(
    const float *restrict q,  int ldq, /* 6 query rows                         */
    const float *restrict kp,          /* panel: d rows of 64 floats           */
    const float *restrict vp, int ldv, /* 64 value rows                        */
    float *restrict oacc,     int dp,  /* 6 rows of dp floats                  */
    __m512 *restrict m,                /* running max per row (base 2, all lanes) */
    __m512 *restrict lvec,             /* running sum per row, 16 partial lanes */
    int d, float c, int nvalid)        /* keys of this panel that exist (<= 64) */
{
    __m512 s[MR][4];
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < 4; v++)
            s[r][v] = _mm512_setzero_ps();

    /* S = q * kp */
    for (int k = 0; k < d; k++) {
        const float *b = kp + (size_t)k * BC;
        const __m512 b0 = _mm512_load_ps(b), b1 = _mm512_load_ps(b + 16);
        const __m512 b2 = _mm512_load_ps(b + 32), b3 = _mm512_load_ps(b + 48);
        for (int r = 0; r < MR; r++) {
            const __m512 a = _mm512_set1_ps(q[r * ldq + k]);
            s[r][0] = _mm512_fmadd_ps(a, b0, s[r][0]);
            s[r][1] = _mm512_fmadd_ps(a, b1, s[r][1]);
            s[r][2] = _mm512_fmadd_ps(a, b2, s[r][2]);
            s[r][3] = _mm512_fmadd_ps(a, b3, s[r][3]);
        }
    }

    if (nvalid < BC) { /* only in the last panel when N % 64 != 0 */
        const __m512 neg = _mm512_set1_ps(NEG_BIG);
        for (int v = 0; v < 4; v++) {
            const int nv = nvalid - 16 * v;
            const __mmask16 keep = nv >= 16 ? 0xFFFF : nv <= 0 ? 0 : (__mmask16)((1u << nv) - 1);
            for (int r = 0; r < MR; r++)
                s[r][v] = _mm512_mask_blend_ps(keep, neg, s[r][v]);
        }
    }

    /* online softmax, all in vector registers. Phases are separated so the six
       independent rows overlap: (1) row maxima, (2) new max and alpha, (3) exps. */
    float P[MR][BC] __attribute__((aligned(64)));
    __m512 alpha[MR], mnew[MR];
    const __m512 cv = _mm512_set1_ps(c);
    for (int r = 0; r < MR; r++)
        mnew[r] = _mm512_max_ps(_mm512_max_ps(s[r][0], s[r][1]), _mm512_max_ps(s[r][2], s[r][3]));
    for (int r = 0; r < MR; r++) /* butterfly: every lane ends up holding the row max */
        mnew[r] = _mm512_max_ps(mnew[r], _mm512_shuffle_f32x4(mnew[r], mnew[r], 0x4E));
    for (int r = 0; r < MR; r++)
        mnew[r] = _mm512_max_ps(mnew[r], _mm512_shuffle_f32x4(mnew[r], mnew[r], 0xB1));
    for (int r = 0; r < MR; r++)
        mnew[r] = _mm512_max_ps(mnew[r], _mm512_permute_ps(mnew[r], 0x4E));
    for (int r = 0; r < MR; r++)
        mnew[r] = _mm512_max_ps(mnew[r], _mm512_permute_ps(mnew[r], 0xB1));
    for (int r = 0; r < MR; r++) {
        mnew[r] = _mm512_max_ps(m[r], _mm512_mul_ps(mnew[r], cv));
        alpha[r] = exp2_ps(_mm512_sub_ps(m[r], mnew[r])); /* 0 on the first panel */
        m[r] = mnew[r];
    }
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < 4; v++)
            s[r][v] = exp2_ps(_mm512_fmsub_ps(s[r][v], cv, mnew[r]));
    for (int r = 0; r < MR; r++) {
        lvec[r] = _mm512_fmadd_ps(lvec[r], alpha[r],
                                  _mm512_add_ps(_mm512_add_ps(s[r][0], s[r][1]), _mm512_add_ps(s[r][2], s[r][3])));
        for (int v = 0; v < 4; v++)
            _mm512_store_ps(&P[r][16 * v], s[r][v]);
    }

    /* O = alpha * O + P * V, 64 output columns at a time */
    for (int c0 = 0; c0 < dp; c0 += 64) {
        __m512 o[MR][4];
        for (int r = 0; r < MR; r++)
            for (int v = 0; v < 4; v++)
                o[r][v] = _mm512_mul_ps(alpha[r], _mm512_load_ps(oacc + r * dp + c0 + 16 * v));
        for (int j = 0; j < BC; j++) {
            const float *b = vp + (size_t)j * ldv + c0;
            const __m512 b0 = _mm512_loadu_ps(b), b1 = _mm512_loadu_ps(b + 16);
            const __m512 b2 = _mm512_loadu_ps(b + 32), b3 = _mm512_loadu_ps(b + 48);
            for (int r = 0; r < MR; r++) {
                const __m512 a = _mm512_set1_ps(P[r][j]);
                o[r][0] = _mm512_fmadd_ps(a, b0, o[r][0]);
                o[r][1] = _mm512_fmadd_ps(a, b1, o[r][1]);
                o[r][2] = _mm512_fmadd_ps(a, b2, o[r][2]);
                o[r][3] = _mm512_fmadd_ps(a, b3, o[r][3]);
            }
        }
        for (int r = 0; r < MR; r++)
            for (int v = 0; v < 4; v++)
                _mm512_store_ps(oacc + r * dp + c0 + 16 * v, o[r][v]);
    }
}

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const int dp = (d + 63) / 64 * 64; /* head dimension padded for the PV tile */
    const int np = (N + BC - 1) / BC;  /* key panels                           */
    const float c = 1.4426950408889634f / sqrtf((float)d); /* log2(e)/sqrt(d) */

    /* K panels (zero-padded keys); V in place if its rows are whole 64-column chunks */
    float *Kp = aligned_alloc(64, (size_t)np * d * BC * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int p = 0; p < np; p++) {
        float *panel = Kp + (size_t)p * d * BC;
        const int nv = N - p * BC < BC ? N - p * BC : BC;
        for (int jj = 0; jj < BC; jj++)
            for (int k = 0; k < d; k++)
                panel[(size_t)k * BC + jj] = jj < nv ? K[(size_t)(p * BC + jj) * d + k] : 0.0f;
    }
    const float *Vuse = V;
    float *Vp = NULL;
    int ldv = d;
    { /* aligned, padded copy: the driver's malloc gives only 16-byte alignment, and
         unaligned 64-byte loads of V would split across cache lines */
        Vp = aligned_alloc(64, (size_t)np * BC * dp * sizeof(float));
#pragma omp parallel for schedule(static)
        for (int j = 0; j < np * BC; j++) {
            memset(Vp + (size_t)j * dp, 0, sizeof(float) * dp);
            if (j < N)
                memcpy(Vp + (size_t)j * dp, V + (size_t)j * d, sizeof(float) * d);
        }
        Vuse = Vp;
        ldv = dp;
    }
    if (d > 512) {
        fprintf(stderr, "attn_omp: d > 512 not supported\n");
        exit(1);
    }
    /* Query block size: about 2 blocks per thread (dynamic scheduling then evens out
       slow or shared cores), at least one strip and at most BR_MAX rows. */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    int br = (N + 2 * nthreads - 1) / (2 * nthreads);
    br = (br + MR - 1) / MR * MR;
    br = br < MR ? MR : br > BR_MAX ? BR_MAX : br;
    const int nblk = (N + br - 1) / br;

#pragma omp parallel
    {
        float *oacc = aligned_alloc(64, (size_t)BR_MAX * dp * sizeof(float));
        float qtail[MR * 512] __attribute__((aligned(64))); /* zero-padded partial last strip */
        __m512 m[BR_MAX], lvec[BR_MAX];

#pragma omp for schedule(dynamic, 1)
        for (int blk = 0; blk < nblk; blk++) {
            const int i0 = blk * br;
            const int rows = N - i0 < br ? N - i0 : br;
            const int nstrips = (rows + MR - 1) / MR;
            for (int r = 0; r < br; r++) {
                m[r] = _mm512_set1_ps(NEG_BIG);
                lvec[r] = _mm512_setzero_ps();
            }
            memset(oacc, 0, (size_t)br * dp * sizeof(float));
            const int last = rows % MR;
            if (last) {
                memset(qtail, 0, sizeof(float) * MR * d);
                memcpy(qtail, Q + (size_t)(i0 + rows - last) * d, sizeof(float) * last * d);
            }

            for (int p = 0; p < np; p++) {
                const int nvalid = N - p * BC < BC ? N - p * BC : BC;
                const float *kp = Kp + (size_t)p * d * BC;
                const float *vp = Vuse + (size_t)p * BC * ldv;
                for (int st = 0; st < nstrips; st++) {
                    const int s = st * MR;
                    const float *q = (last && st == nstrips - 1) ? qtail : Q + (size_t)(i0 + s) * d;
                    strip_panel(q, d, kp, vp, ldv, oacc + (size_t)s * dp, dp, m + s, lvec + s, d, c, nvalid);
                }
            }

            for (int r = 0; r < rows; r++) {
                const float inv = 1.0f / _mm512_reduce_add_ps(lvec[r]);
                for (int k = 0; k < d; k++)
                    O[(size_t)(i0 + r) * d + k] = oacc[(size_t)r * dp + k] * inv;
            }
        }
        free(oacc);
    }
    free(Vp);
    free(Kp);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
