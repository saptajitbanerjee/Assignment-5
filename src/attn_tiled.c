/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Still the three separate steps of attn_naive.c (S is stored), but both matrix
 * products use a register-tiled AVX-512 micro-kernel plus cache blocking:
 *
 *   micro-kernel  C[6 x 64] (+)= A[6 x kdim] * B[kdim x 64]
 *     24 zmm accumulators (6 rows x 4 vectors of 16 floats) stay in registers
 *     for the whole k loop. Per k: 4 vector loads of B, 6 broadcasts of A,
 *     24 FMAs. Each loaded B value is reused 6 times and each A value 64 times.
 *
 *   1. S = Q K^T: K is packed once into panels Kp[p][k][0..63] (64 keys per
 *      panel, contiguous, zero-padded), so B rows are contiguous. Loop order:
 *      row block (MB rows of Q, kept in L2) -> key panel (16-32 KB, L1) -> 6-row strip.
 *   2. softmax: as in attn_loops.c (scalar expf).
 *   3. O = S V: rows of S are the A operand, V (padded to a multiple of 64
 *      columns) is B. Loop order: row block (O block in L2) -> 64-key block of V
 *      (L1) -> 6-row strip -> 64-column chunk of O.
 *
 * Build with -DPROFILE to print the time of each step to stderr.
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

#define MR 6        /* rows per register tile          */
#define NR 64       /* columns per register tile (4 zmm) */
#define MB 192      /* rows per cache block (multiple of MR) */
#define KC 64       /* keys per V block in step 3       */

#ifdef PROFILE
static double tnow(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif

/* C[6][64] (ldc) = (accumulate ? C : 0) + alpha * A[6][kdim] (lda) * B[kdim][64] (ldb).
 * alpha is applied once at the end (only used without accumulate). */
static inline __attribute__((always_inline)) void kernel_6x64(int kdim, const float *restrict A, size_t lda,
                                                              const float *restrict B, size_t ldb,
                                                              float *restrict C, size_t ldc,
                                                              int accumulate, float alpha)
{
    __m512 c[MR][4];
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < 4; v++)
            c[r][v] = accumulate ? _mm512_loadu_ps(C + r * ldc + 16 * v) : _mm512_setzero_ps();
    for (int k = 0; k < kdim; k++) {
        const float *b = B + (size_t)k * ldb;
        const __m512 b0 = _mm512_loadu_ps(b), b1 = _mm512_loadu_ps(b + 16);
        const __m512 b2 = _mm512_loadu_ps(b + 32), b3 = _mm512_loadu_ps(b + 48);
        for (int r = 0; r < MR; r++) {
            const __m512 a = _mm512_set1_ps(A[r * lda + k]);
            c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
            c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
            c[r][2] = _mm512_fmadd_ps(a, b2, c[r][2]);
            c[r][3] = _mm512_fmadd_ps(a, b3, c[r][3]);
        }
    }
    const __m512 al = _mm512_set1_ps(alpha);
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < 4; v++)
            _mm512_storeu_ps(C + r * ldc + 16 * v, accumulate ? c[r][v] : _mm512_mul_ps(c[r][v], al));
}

/* Same as kernel_6x64 but for a partial tile (mr <= 6 rows, nc <= 64 columns):
 * copies A rows and C into zero-padded buffers, so B must have 64 readable columns. */
static void tile_edge(int kdim, const float *A, size_t lda, int mr, const float *B, size_t ldb,
                      float *C, size_t ldc, int nc, int accumulate, float alpha, float *abuf)
{
    float cbuf[MR * NR];
    memset(abuf, 0, sizeof(float) * MR * (size_t)kdim);
    memset(cbuf, 0, sizeof cbuf);
    for (int r = 0; r < mr; r++) {
        memcpy(abuf + (size_t)r * kdim, A + r * lda, sizeof(float) * kdim);
        if (accumulate)
            memcpy(cbuf + r * NR, C + r * ldc, sizeof(float) * nc);
    }
    kernel_6x64(kdim, abuf, kdim, B, ldb, cbuf, NR, accumulate, alpha);
    for (int r = 0; r < mr; r++)
        memcpy(C + r * ldc, cbuf + r * NR, sizeof(float) * nc);
}

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int np = (N + NR - 1) / NR;      /* key panels                    */
    const int dp = (d + NR - 1) / NR * NR; /* d padded to a multiple of 64  */
    float *S = aligned_alloc(64, (size_t)N * N * sizeof(float));
    float *Kp = aligned_alloc(64, (size_t)np * d * NR * sizeof(float));
    float *Vp = aligned_alloc(64, ((size_t)N + 1) * dp * sizeof(float));
    float *abuf = aligned_alloc(64, sizeof(float) * MR * (size_t)(d > KC ? d : KC));
#ifdef PROFILE
    double t0 = tnow();
#endif

    /* 0. pack K into 64-key panels (transposed) and V into rows of dp floats */
    for (int p = 0; p < np; p++)
        for (int k = 0; k < d; k++)
            for (int jj = 0; jj < NR; jj++) {
                const int j = p * NR + jj;
                Kp[((size_t)p * d + k) * NR + jj] = j < N ? K[(size_t)j * d + k] : 0.0f;
            }
    memset(Vp, 0, ((size_t)N + 1) * dp * sizeof(float));
    for (int j = 0; j < N; j++)
        memcpy(Vp + (size_t)j * dp, V + (size_t)j * d, sizeof(float) * d);

    /* 1. S = Q K^T * scale */
    for (int i0 = 0; i0 < N; i0 += MB) {
        const int i1 = i0 + MB < N ? i0 + MB : N;
        for (int p = 0; p < np; p++) {
            const float *B = Kp + (size_t)p * d * NR;
            const int nc = N - p * NR < NR ? N - p * NR : NR;
            for (int i = i0; i < i1; i += MR) {
                const int mr = i1 - i < MR ? i1 - i : MR;
                float *C = S + (size_t)i * N + (size_t)p * NR;
                if (mr == MR && nc == NR)
                    kernel_6x64(d, Q + (size_t)i * d, d, B, NR, C, N, 0, scale);
                else
                    tile_edge(d, Q + (size_t)i * d, d, mr, B, NR, C, N, nc, 0, scale, abuf);
            }
        }
    }
#ifdef PROFILE
    double t1 = tnow();
#endif

    /* 2. softmax of each row of S (as in attn_loops.c) */
    for (int i = 0; i < N; i++) {
        float *restrict row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            max = row[j] > max ? row[j] : max;
        for (int j = 0; j < N; j++)
            row[j] = expf(row[j] - max);
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += row[j];
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }
#ifdef PROFILE
    double t2 = tnow();
#endif

    /* 3. O = S V */
    for (int i0 = 0; i0 < N; i0 += MB) {
        const int i1 = i0 + MB < N ? i0 + MB : N;
        for (int j0 = 0; j0 < N; j0 += KC) {
            const int kc = N - j0 < KC ? N - j0 : KC;
            for (int i = i0; i < i1; i += MR) {
                const int mr = i1 - i < MR ? i1 - i : MR;
                const float *A = S + (size_t)i * N + j0;
                for (int c0 = 0; c0 < d; c0 += NR) {
                    const int nc = d - c0 < NR ? d - c0 : NR;
                    const float *B = Vp + (size_t)j0 * dp + c0;
                    float *C = O + (size_t)i * d + c0;
                    if (j0 == 0) /* first key block: start from zero */
                        for (int r = 0; r < mr; r++)
                            memset(C + (size_t)r * d, 0, sizeof(float) * nc);
                    if (mr == MR && nc == NR)
                        kernel_6x64(kc, A, N, B, dp, C, d, 1, 1.0f);
                    else
                        tile_edge(kc, A, N, mr, B, dp, C, d, nc, 1, 1.0f, abuf);
                }
            }
        }
    }
#ifdef PROFILE
    double t3 = tnow();
    fprintf(stderr, "[tiled N=%d d=%d] pack %.1f ms | QK^T %.1f ms | softmax %.1f ms | SV %.1f ms\n", N, d,
            0.0, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3);
#endif

    free(abuf);
    free(Vp);
    free(Kp);
    free(S);
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
