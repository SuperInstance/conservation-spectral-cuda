/*
 * bench_scale.cu — Scale benchmark for Conservation Spectral CUDA
 *
 * Benchmarks GPU vs CPU performance on graphs from 100 to 10000 nodes.
 * Measures:
 *   - Laplacian construction time
 *   - Eigendecomposition time
 *   - Full analysis (Laplacian + Eigen + Conservation) time
 *   - Spectral gap computation
 *
 * Run: ./bench_scale [--max-n N]
 */

#include "conservation_spectral_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
 * Timing helpers
 * ============================================================ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* CUDA event timing */
typedef struct {
    cudaEvent_t start, stop;
} cuda_timer_t;

static void timer_create(cuda_timer_t* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

static void timer_destroy(cuda_timer_t* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static void timer_begin(cuda_timer_t* t) {
    cudaEventRecord(t->start);
}

static float timer_end(cuda_timer_t* t) {
    cudaEventRecord(t->stop);
    cudaEventSynchronize(t->stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

/* ============================================================
 * CPU reference implementation (for comparison)
 * ============================================================ */

/* CPU Laplacian construction */
static double cpu_build_laplacian(const float* W, float* L, int n, bool normalized) {
    double t0 = now_sec();

    float* degree = (float*)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            degree[i] += W[i * n + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (normalized) {
                float di = (degree[i] > 1e-12f) ? 1.0f / sqrtf(degree[i]) : 0.0f;
                float dj = (degree[j] > 1e-12f) ? 1.0f / sqrtf(degree[j]) : 0.0f;
                float val = di * W[i * n + j] * dj;
                L[i * n + j] = (i == j) ? 1.0f - val : -val;
            } else {
                L[i * n + j] = (i == j) ? degree[i] - W[i * n + j] : -W[i * n + j];
            }
        }
    }

    free(degree);
    return now_sec() - t0;
}

/* CPU power iteration for top-k eigenvalues */
static double cpu_eigendecompose(const float* L, float* eigenvalues, int n, int k) {
    double t0 = now_sec();

    /* Find shift */
    float shift = 0.0f;
    for (int i = 0; i < n; i++) {
        if (L[i * n + i] > shift) shift = L[i * n + i];
    }

    /* Build M = shift*I - L */
    float* M = (float*)malloc(n * n * sizeof(float));
    float* R = (float*)malloc(n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) M[i] = -L[i];
    for (int i = 0; i < n; i++) M[i * n + i] += shift;
    memcpy(R, M, n * n * sizeof(float));

    float* v = (float*)malloc(n * sizeof(float));
    float* w = (float*)malloc(n * sizeof(float));

    for (int ev = 0; ev < k; ev++) {
        for (int i = 0; i < n; i++) v[i] = 1.0f / (float)(i + 1 + ev * 3);

        float lambda = 0.0f;
        for (int iter = 0; iter < 1000; iter++) {
            /* w = R * v */
            for (int i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) sum += R[i * n + j] * v[j];
                w[i] = sum;
            }

            float norm = 0.0f;
            for (int i = 0; i < n; i++) norm += w[i] * w[i];
            norm = sqrtf(norm);
            if (norm < 1e-30f) break;
            for (int i = 0; i < n; i++) v[i] = w[i] / norm;

            /* Rayleigh quotient */
            float rq = 0.0f;
            for (int i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) sum += R[i * n + j] * v[j];
                rq += v[i] * sum;
            }

            if (fabsf(rq - lambda) < 1e-8f) { lambda = rq; break; }
            lambda = rq;
        }

        eigenvalues[ev] = shift - lambda;

        /* Deflate */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                R[i * n + j] -= lambda * v[i] * v[j];
            }
        }
    }

    free(M); free(R); free(v); free(w);
    return now_sec() - t0;
}

/* ============================================================
 * Graph generators
 * ============================================================ */

/* Generate a random graph with given average degree.
 * Fills W as adjacency weight matrix. */
static void generate_random_graph(float* W, int n, float avg_degree) {
    memset(W, 0, n * n * sizeof(float));
    float prob = avg_degree / (float)n;
    unsigned int seed = 42;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            /* Simple LCG random */
            seed = seed * 1103515245 + 12345;
            float r = (float)((seed >> 16) & 0x7fff) / 32768.0f;
            if (r < prob) {
                float w = 0.5f + r; /* weight in [0.5, 1.5) */
                W[i * n + j] = w;
                W[j * n + i] = w;
            }
        }
    }
}

/* Generate a chain/path graph */
static void generate_chain_graph(float* W, int n) {
    memset(W, 0, n * n * sizeof(float));
    for (int i = 0; i < n - 1; i++) {
        W[i * n + (i + 1)] = 1.0f;
        W[(i + 1) * n + i] = 1.0f;
    }
}

/* ============================================================
 * Benchmark runner
 * ============================================================ */

typedef struct {
    int n;
    float gpu_laplacian_ms;
    float gpu_eigen_ms;
    float gpu_analysis_ms;
    double cpu_laplacian_sec;
    double cpu_eigen_sec;
    double cpu_analysis_sec;
    float spectral_gap;
    float speedup_laplacian;
    float speedup_eigen;
    float speedup_analysis;
} bench_result;

static void run_benchmark(int n, int k, bench_result* result) {
    result->n = n;

    size_t mat_bytes = (size_t)n * n * sizeof(float);
    float* W = (float*)malloc(mat_bytes);
    float* L_cpu = (float*)malloc(mat_bytes);
    float* evals_cpu = (float*)malloc(k * sizeof(float));
    float* h_ratios = (float*)malloc(n * sizeof(float));
    float* h_attrs = (float*)malloc(n * sizeof(float));

    generate_random_graph(W, n, 6.0f);
    for (int i = 0; i < n; i++) h_attrs[i] = (float)i * 0.1f;

    cuda_timer_t gpu_timer;
    timer_create(&gpu_timer);

    /* --- GPU Laplacian --- */
    csc_cuda_context* ctx = csc_cuda_create(n);
    timer_begin(&gpu_timer);
    csc_cuda_build_laplacian(ctx, W, false);
    result->gpu_laplacian_ms = timer_end(&gpu_timer);

    /* --- CPU Laplacian --- */
    result->cpu_laplacian_sec = cpu_build_laplacian(W, L_cpu, n, false);

    /* --- GPU Eigen --- */
    timer_begin(&gpu_timer);
    if (n <= 512) {
        csc_cuda_eigendecompose(ctx);
    } else {
        csc_cuda_eigendecompose_partial(ctx, k);
    }
    result->gpu_eigen_ms = timer_end(&gpu_timer);

    /* --- CPU Eigen (only for smaller graphs) --- */
    if (n <= 2000) {
        result->cpu_eigen_sec = cpu_eigendecompose(L_cpu, evals_cpu, n, k);
    } else {
        result->cpu_eigen_sec = -1.0; /* skip: too slow */
    }

    /* --- GPU Full Analysis --- */
    /* Re-create for clean timing */
    csc_cuda_destroy(ctx);
    ctx = csc_cuda_create(n);

    timer_begin(&gpu_timer);
    csc_cuda_build_laplacian(ctx, W, false);
    if (n <= 512) {
        csc_cuda_eigendecompose(ctx);
    } else {
        csc_cuda_eigendecompose_partial(ctx, k);
    }
    csc_cuda_conservation_ratio(ctx, h_attrs, h_ratios);
    result->gpu_analysis_ms = timer_end(&gpu_timer);

    result->spectral_gap = csc_cuda_spectral_gap(ctx);

    /* CPU full analysis */
    double t0 = now_sec();
    cpu_build_laplacian(W, L_cpu, n, false);
    if (n <= 2000) {
        cpu_eigendecompose(L_cpu, evals_cpu, n, k);
    }
    result->cpu_analysis_sec = (n <= 2000) ? (now_sec() - t0) : -1.0;

    /* Compute speedups */
    result->speedup_laplacian = (float)(result->cpu_laplacian_sec * 1000.0 / result->gpu_laplacian_ms);
    if (result->cpu_eigen_sec > 0) {
        result->speedup_eigen = (float)(result->cpu_eigen_sec * 1000.0 / result->gpu_eigen_ms);
    } else {
        result->speedup_eigen = -1.0f;
    }
    if (result->cpu_analysis_sec > 0) {
        result->speedup_analysis = (float)(result->cpu_analysis_sec * 1000.0 / result->gpu_analysis_ms);
    } else {
        result->speedup_analysis = -1.0f;
    }

    timer_destroy(&gpu_timer);
    csc_cuda_destroy(ctx);

    free(W); free(L_cpu); free(evals_cpu); free(h_ratios); free(h_attrs);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    int max_n = 10000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-n") == 0 && i + 1 < argc) {
            max_n = atoi(argv[i + 1]);
        }
    }

    printf("=== Conservation Spectral CUDA — Scale Benchmark ===\n\n");
    csc_cuda_print_device_info();
    printf("\n");

    int sizes[] = {100, 250, 500, 1000, 2500, 5000, 10000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s  %-12s  %-12s  %-12s  %-8s  %-8s  %-8s\n",
           "Nodes", "GPU Lap(ms)", "GPU Eigen(ms)", "GPU Full(ms)",
           "Spd Lap", "Spd Eigen", "Spd Full");
    printf("%-8s  %-12s  %-12s  %-12s  %-8s  %-8s  %-8s\n",
           "-----", "-----------", "------------", "-----------",
           "-------", "--------", "-------");

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        if (n > max_n) break;

        int k = (n <= 512) ? n : 10;

        bench_result res;
        memset(&res, 0, sizeof(res));
        run_benchmark(n, k, &res);

        char spd_lap[32], spd_eig[32], spd_full[32];
        snprintf(spd_lap, sizeof(spd_lap), "%.1fx", res.speedup_laplacian);
        snprintf(spd_eig, sizeof(spd_eig), "%s",
                 res.speedup_eigen > 0 ? "" : "skip");
        if (res.speedup_eigen > 0) snprintf(spd_eig, sizeof(spd_eig), "%.1fx", res.speedup_eigen);
        snprintf(spd_full, sizeof(spd_full), "%s",
                 res.speedup_analysis > 0 ? "" : "skip");
        if (res.speedup_analysis > 0) snprintf(spd_full, sizeof(spd_full), "%.1fx", res.speedup_analysis);

        printf("%-8d  %-12.2f  %-12.2f  %-12.2f  %-8s  %-8s  %-8s\n",
               n,
               res.gpu_laplacian_ms,
               res.gpu_eigen_ms,
               res.gpu_analysis_ms,
               spd_lap, spd_eig, spd_full);
    }

    printf("\nDone.\n");
    return 0;
}
