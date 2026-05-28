/*
 * tracker.cu — Real-time conservation tracker with GPU-accelerated updates
 *
 * Sliding window anomaly detection using z-score thresholds.
 * GPU kernels for parallel window update and statistics.
 */

#include "conservation_spectral_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * CUDA Kernels
 * ============================================================ */

/* Shift sliding window left by 1 and append new value.
 * Each thread shifts one element. */
__global__ void tracker_update_kernel(float* __restrict__ history,
                                       float observation,
                                       int window_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= window_size - 1) return;
    history[i] = history[i + 1];

    /* Last thread writes the new value */
    if (i == window_size - 2) {
        history[window_size - 1] = observation;
    }
}

/* Compute mean of the window. Single-thread reduction. */
__global__ void tracker_mean_kernel(const float* __restrict__ history,
                                     float* __restrict__ mean_out,
                                     int window_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < window_size; i++) {
            sum += history[i];
        }
        *mean_out = sum / (float)window_size;
    }
}

/* Compute standard deviation given mean. Single-thread reduction. */
__global__ void tracker_std_kernel(const float* __restrict__ history,
                                    float mean,
                                    float* __restrict__ std_out,
                                    int window_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float var = 0.0f;
        for (int i = 0; i < window_size; i++) {
            float d = history[i] - mean;
            var += d * d;
        }
        *std_out = sqrtf(var / (float)window_size);
    }
}

/* Compute z-score of latest value and classify */
__global__ void tracker_zscore_kernel(const float* __restrict__ history,
                                       float mean, float std,
                                       int* __restrict__ alert_out,
                                       int window_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float latest = history[window_size - 1];
        if (std < 1e-12f) {
            *alert_out = 0;
            return;
        }
        float zscore = fabsf(latest - mean) / std;
        if (zscore > 3.0f) {
            *alert_out = 2;  /* critical */
        } else if (zscore > 2.0f) {
            *alert_out = 1;  /* warning */
        } else {
            *alert_out = 0;  /* nominal */
        }
    }
}

/* Initialize history on device with a value */
__global__ void tracker_init_kernel(float* __restrict__ history,
                                     int window_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= window_size) return;
    history[i] = 0.0f;
}

/* ============================================================
 * Tracker API
 * ============================================================ */

CscCudaError csc_cuda_tracker_init(csc_cuda_context* ctx, int window_size) {
    if (!ctx) return CSC_CUDA_ERR_NULL_PTR;
    if (window_size <= 0) return CSC_CUDA_ERR_DIMENSION;

    /* Free old history if any */
    if (ctx->d_tracker_history) {
        cudaFree(ctx->d_tracker_history);
    }

    cudaError_t err = cudaMalloc(&ctx->d_tracker_history,
                                  window_size * sizeof(float));
    if (err != cudaSuccess) return CSC_CUDA_ERR_ALLOC;

    /* Zero-initialize */
    int block = 256;
    int grid = (window_size + block - 1) / block;
    tracker_init_kernel<<<grid, block>>>(ctx->d_tracker_history, window_size);
    cudaDeviceSynchronize();

    ctx->tracker_window_size = window_size;
    ctx->tracker_count = 0;
    ctx->tracker_baseline_set = false;
    ctx->baseline_mean = 0.0f;
    ctx->baseline_std = 0.0f;

    return CSC_CUDA_OK;
}

int csc_cuda_tracker_feed(csc_cuda_context* ctx, float observation) {
    if (!ctx || !ctx->d_tracker_history) return 0;

    int ws = ctx->tracker_window_size;
    int block = 256;

    if (ctx->tracker_count < ws) {
        /* Still filling the window — write directly */
        float* d_slot = ctx->d_tracker_history + ctx->tracker_count;
        cudaMemcpy(d_slot, &observation, sizeof(float), cudaMemcpyHostToDevice);
        ctx->tracker_count++;
    } else {
        /* Shift window left, append new value */
        int grid = (ws - 1 + block - 1) / block;
        tracker_update_kernel<<<grid, block>>>(ctx->d_tracker_history,
                                                observation, ws);
        cudaDeviceSynchronize();
    }

    /* Establish baseline after first full window */
    if (ctx->tracker_count == ws && !ctx->tracker_baseline_set) {
        float* d_mean = NULL;
        float* d_std = NULL;
        cudaMalloc(&d_mean, sizeof(float));
        cudaMalloc(&d_std, sizeof(float));

        tracker_mean_kernel<<<1, 1>>>(ctx->d_tracker_history, d_mean, ws);
        cudaDeviceSynchronize();
        cudaMemcpy(&ctx->baseline_mean, d_mean, sizeof(float),
                   cudaMemcpyDeviceToHost);

        tracker_std_kernel<<<1, 1>>>(ctx->d_tracker_history,
                                      ctx->baseline_mean, d_std, ws);
        cudaDeviceSynchronize();
        cudaMemcpy(&ctx->baseline_std, d_std, sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_mean);
        cudaFree(d_std);

        ctx->tracker_baseline_set = true;
        return 0; /* Just established baseline */
    }

    return csc_cuda_tracker_check(ctx);
}

int csc_cuda_tracker_check(const csc_cuda_context* ctx) {
    if (!ctx || !ctx->tracker_baseline_set || ctx->tracker_count == 0) return 0;

    /* Compute z-score on GPU */
    float* d_mean = NULL;
    float* d_std = NULL;
    int* d_alert = NULL;

    cudaError_t err;
    err = cudaMalloc(&d_mean, sizeof(float));
    if (err != cudaSuccess) return 0;
    err = cudaMalloc(&d_std, sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_mean); return 0; }
    err = cudaMalloc(&d_alert, sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_mean); cudaFree(d_std); return 0; }

    /* Recompute current statistics */
    float mean = ctx->baseline_mean;
    float std = ctx->baseline_std;
    cudaMemcpy(d_mean, &mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std, &std, sizeof(float), cudaMemcpyHostToDevice);

    tracker_zscore_kernel<<<1, 1>>>((const float*)ctx->d_tracker_history,
                                     mean, std, d_alert,
                                     ctx->tracker_window_size);
    cudaDeviceSynchronize();

    int alert = 0;
    cudaMemcpy(&alert, d_alert, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_mean);
    cudaFree(d_std);
    cudaFree(d_alert);

    return alert;
}
