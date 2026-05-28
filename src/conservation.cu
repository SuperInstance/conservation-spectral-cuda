/*
 * conservation.cu — Conservation analysis on GPU
 *
 * Kernels for conservation ratios, spectral gap, Cheeger constant.
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

/* Project attribute onto eigenvector: projection[i] = attr[i] * evec[i]
 * evec is column col_idx of column-major eigenvector matrix */
__global__ void project_kernel(const float* __restrict__ attributes,
                                const float* __restrict__ eigenvectors,
                                float* __restrict__ projection,
                                int col_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    projection[i] = attributes[i] * eigenvectors[col_idx * n + i];
}

/* Compute gradient: gradient[i] = projection[i+1] - projection[i], length n-1 */
__global__ void gradient_kernel(const float* __restrict__ projection,
                                 float* __restrict__ gradient,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;
    gradient[i] = projection[i + 1] - projection[i];
}

/* Compute variance of an array. Single-thread reduction. */
__global__ void variance_kernel(const float* __restrict__ data,
                                 float* __restrict__ result,
                                 int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += data[i];
        mean /= (float)n;

        float var = 0.0f;
        for (int i = 0; i < n; i++) {
            float d = data[i] - mean;
            var += d * d;
        }
        *result = var / (float)n;
    }
}

/* Compute spectral gap: max gap between consecutive eigenvalues.
 * Single-thread reduction. */
__global__ void spectral_gap_kernel(const float* __restrict__ eigenvalues,
                                     float* __restrict__ result,
                                     int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_gap = 0.0f;
        for (int i = 0; i < n - 1; i++) {
            float gap = eigenvalues[i + 1] - eigenvalues[i];
            if (gap > max_gap) max_gap = gap;
        }
        *result = max_gap;
    }
}

/* Cheeger constant approximation from Fiedler vector.
 * Partitions vertices by Fiedler vector sign, computes cut/volume ratio.
 * Each block handles a chunk of rows. */
__global__ void cheeger_kernel(const float* __restrict__ laplacian,
                                const float* __restrict__ fiedler,
                                float* __restrict__ cut_buf,
                                float* __restrict__ vol_s_buf,
                                int n) {
    extern __shared__ float sdata[];  /* 2*blockDim.x floats */

    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float local_cut = 0.0f;
    float local_vol = 0.0f;

    if (row < n && fiedler[row] < 0.0f) {
        /* This vertex is in S */
        for (int j = 0; j < n; j++) {
            float w = -laplacian[row * n + j];
            if (row != j) {
                local_vol += fabsf(w);
                if (fiedler[j] >= 0.0f) {
                    local_cut += fabsf(w);
                }
            }
        }
    }

    /* Shared memory reduction */
    sdata[tid] = local_cut;
    sdata[blockDim.x + tid] = local_vol;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        cut_buf[blockIdx.x] = sdata[0];
        vol_s_buf[blockIdx.x] = sdata[blockDim.x];
    }
}

/* Final Cheeger reduction: sum partial results and compute ratio */
__global__ void cheeger_final_kernel(const float* __restrict__ cut_buf,
                                      const float* __restrict__ vol_s_buf,
                                      float* __restrict__ result,
                                      float total_vol,
                                      int n_blocks) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float cut = 0.0f;
        float vol_s = 0.0f;
        for (int i = 0; i < n_blocks; i++) {
            cut += cut_buf[i];
            vol_s += vol_s_buf[i];
        }
        float vol_comp = total_vol - vol_s;
        float min_vol = (vol_s < vol_comp) ? vol_s : vol_comp;
        *result = (min_vol > 1e-12f) ? (cut / min_vol) : 0.0f;
    }
}

/* Compute total volume (sum of diagonal of Laplacian) */
__global__ void total_volume_kernel(const float* __restrict__ laplacian,
                                     float* __restrict__ result,
                                     int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float vol = 0.0f;
        for (int i = 0; i < n; i++) {
            vol += laplacian[i * n + i];
        }
        *result = vol;
    }
}

/* ============================================================
 * Conservation ratio (full)
 * ============================================================ */

CscCudaError csc_cuda_conservation_ratio(csc_cuda_context* ctx,
                                          const float* h_attributes,
                                          float* h_ratios) {
    if (!ctx || !h_attributes || !h_ratios) return CSC_CUDA_ERR_NULL_PTR;
    if (!ctx->eigen_computed) return CSC_CUDA_ERR_INVALID_STATE;

    int n = ctx->n;
    int k = ctx->k;
    int block = 256;
    int grid = (n + block - 1) / block;

    size_t vec_bytes = (size_t)n * sizeof(float);

    /* Allocate device temporaries */
    float* d_attr = NULL;
    float* d_proj = NULL;
    float* d_grad = NULL;
    float* d_var = NULL;

    cudaError_t err;
    err = cudaMalloc(&d_attr, vec_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_proj, vec_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_grad, vec_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_var, sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    /* Copy attributes to device */
    cudaMemcpy(d_attr, h_attributes, vec_bytes, cudaMemcpyHostToDevice);

    /* Compute ratio for each eigenvector */
    for (int ev = 0; ev < k; ev++) {
        /* Project: proj[i] = attr[i] * evec[ev][i] */
        project_kernel<<<grid, block>>>(d_attr, ctx->d_eigenvectors,
                                         d_proj, ev, n);
        cudaDeviceSynchronize();

        /* Gradient */
        int grid_n1 = ((n - 1) + block - 1) / block;
        gradient_kernel<<<grid_n1, block>>>(d_proj, d_grad, n);
        cudaDeviceSynchronize();

        /* Variance of gradient */
        variance_kernel<<<1, 1>>>(d_grad, d_var, n - 1);
        cudaDeviceSynchronize();

        /* Copy result */
        cudaMemcpy(&h_ratios[ev], d_var, sizeof(float), cudaMemcpyDeviceToHost);
    }

cleanup:
    if (d_attr) cudaFree(d_attr);
    if (d_proj) cudaFree(d_proj);
    if (d_grad) cudaFree(d_grad);
    if (d_var) cudaFree(d_var);

    return CSC_CUDA_OK;
}

/* ============================================================
 * Spectral gap
 * ============================================================ */

float csc_cuda_spectral_gap(csc_cuda_context* ctx) {
    if (!ctx || !ctx->eigen_computed) return -1.0f;

    float* d_result = NULL;
    cudaError_t err = cudaMalloc(&d_result, sizeof(float));
    if (err != cudaSuccess) return -1.0f;

    spectral_gap_kernel<<<1, 1>>>(ctx->d_eigenvalues, d_result, ctx->k);
    cudaDeviceSynchronize();

    float gap = 0.0f;
    cudaMemcpy(&gap, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return gap;
}

/* ============================================================
 * Cheeger constant
 * ============================================================ */

float csc_cuda_cheeger_constant(csc_cuda_context* ctx) {
    if (!ctx || !ctx->eigen_computed || ctx->k < 2) return -1.0f;

    int n = ctx->n;
    int block = 256;
    int grid = (n + block - 1) / block;
    int n_blocks = grid;

    /* Get Fiedler vector = eigenvector index 1 */
    float* d_fiedler = ctx->d_eigenvectors + 1 * n; /* column 1 in col-major */

    float total_vol = 0.0f;
    size_t shared_bytes = 2 * block * sizeof(float);
    float cheeger = 0.0f;

    float* d_cut_buf = NULL;
    float* d_vol_buf = NULL;
    float* d_result = NULL;
    float* d_total_vol = NULL;

    cudaError_t err;
    err = cudaMalloc(&d_cut_buf, n_blocks * sizeof(float));
    if (err != cudaSuccess) goto cheeger_cleanup;
    err = cudaMalloc(&d_vol_buf, n_blocks * sizeof(float));
    if (err != cudaSuccess) goto cheeger_cleanup;
    err = cudaMalloc(&d_result, sizeof(float));
    if (err != cudaSuccess) goto cheeger_cleanup;
    err = cudaMalloc(&d_total_vol, sizeof(float));
    if (err != cudaSuccess) goto cheeger_cleanup;

    /* Compute total volume */
    total_volume_kernel<<<1, 1>>>(ctx->d_laplacian, d_total_vol, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&total_vol, d_total_vol, sizeof(float), cudaMemcpyDeviceToHost);

    /* Compute partial cut and volume */
    cheeger_kernel<<<grid, block, shared_bytes>>>(ctx->d_laplacian, d_fiedler,
                                                    d_cut_buf, d_vol_buf, n);
    cudaDeviceSynchronize();

    /* Final reduction */
    cheeger_final_kernel<<<1, 1>>>(d_cut_buf, d_vol_buf, d_result,
                                    total_vol, n_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(&cheeger, d_result, sizeof(float), cudaMemcpyDeviceToHost);

cheeger_cleanup:
    if (d_cut_buf) cudaFree(d_cut_buf);
    if (d_vol_buf) cudaFree(d_vol_buf);
    if (d_result) cudaFree(d_result);
    if (d_total_vol) cudaFree(d_total_vol);

    return cheeger;
}

/* ============================================================
 * Spectral fingerprint
 * ============================================================ */

/* Helper kernel: hash eigenvalues on device */
__global__ void fingerprint_hash_kernel(const float* __restrict__ eigenvalues,
                                          uint64_t* __restrict__ hashes,
                                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* Quantize float bits and hash */
    uint64_t bits = 0;
    /* Reinterpret float bits — use atomicAdd trick for bit cast */
    memcpy(&bits, &eigenvalues[i], sizeof(float));
    /* Extend to 64-bit with mixing */
    bits = ((uint64_t)(uint32_t)bits) | (bits << 32);
    bits ^= (bits >> 33);
    bits *= 0xff51afd7ed558ccdULL;
    bits ^= (bits >> 33);
    bits *= 0xc4ceb9fe1a85ec53ULL;
    bits ^= (bits >> 33);
    hashes[i] = bits;
}

char* csc_cuda_fingerprint_compute(csc_cuda_context* ctx) {
    if (!ctx || !ctx->eigen_computed) return NULL;

    int n = ctx->k;

    /* Copy eigenvalues to host */
    float* h_evals = (float*)malloc(n * sizeof(float));
    if (!h_evals) return NULL;
    cudaMemcpy(h_evals, ctx->d_eigenvalues, n * sizeof(float),
               cudaMemcpyDeviceToHost);

    /* Build hex fingerprint on host */
    size_t hex_len = (size_t)n * 16 + 1;
    char* hex = (char*)calloc(hex_len, sizeof(char));
    if (!hex) { free(h_evals); return NULL; }

    static const char hx[] = "0123456789abcdef";
    size_t pos = 0;

    for (int i = 0; i < n && pos < hex_len - 17; i++) {
        /* Hash each eigenvalue */
        uint64_t bits = 0;
        /* Extend float to 64 bits for mixing */
        uint32_t fbits = 0;
        memcpy(&fbits, &h_evals[i], sizeof(float));
        bits = ((uint64_t)fbits) | ((uint64_t)fbits << 32);
        bits ^= (bits >> 33);
        bits *= 0xff51afd7ed558ccdULL;
        bits ^= (bits >> 33);
        bits *= 0xc4ceb9fe1a85ec53ULL;
        bits ^= (bits >> 33);

        for (int j = 15; j >= 0 && pos < hex_len - 1; j--) {
            hex[pos++] = hx[(bits >> (j * 4)) & 0xF];
        }
    }
    hex[pos] = '\0';

    free(h_evals);
    return hex;
}

float csc_cuda_fingerprint_compare(const char* fp1, const char* fp2) {
    if (!fp1 || !fp2) return 0.0f;

    size_t len1 = strlen(fp1);
    size_t len2 = strlen(fp2);
    if (len1 == 0 && len2 == 0) return 1.0f;

    size_t min_len = (len1 < len2) ? len1 : len2;
    size_t max_len = (len1 > len2) ? len1 : len2;
    if (max_len == 0) return 1.0f;

    size_t matches = 0;
    for (size_t i = 0; i < min_len; i++) {
        if (fp1[i] == fp2[i]) matches++;
    }

    return (float)matches / (float)max_len;
}
