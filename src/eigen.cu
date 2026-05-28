/*
 * eigen.cu — GPU Eigendecomposition for Conservation Spectral CUDA
 *
 * Uses cuSOLVER Dn for full eigendecomposition (dsyevd).
 * Provides power iteration + deflation for partial decomposition.
 */

#include "conservation_spectral_cuda.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * CUDA Kernels
 * ============================================================ */

/* Dense matrix-vector multiply: y = A * x
 * A: n×n row-major, x: n, y: n
 * Each thread computes one element of y. */
__global__ void matvec_kernel(const float* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += A[row * n + j] * x[j];
    }
    y[row] = sum;
}

/* Power iteration: compute w = R * v, then normalize v = w / ||w||
 * Also computes Rayleigh quotient: lambda = v^T * (R*v)
 * R: n×n row-major residual matrix
 * v: current eigenvector estimate (n)
 * w: workspace (n)
 * rq_out: Rayleigh quotient output (1 element on device)
 * norm_out: norm output (1 element on device) */
__global__ void power_iteration_kernel(const float* __restrict__ R,
                                        float* __restrict__ v,
                                        float* __restrict__ w,
                                        int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    /* w[row] = R[row,:] · v */
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += R[row * n + j] * v[j];
    }
    w[row] = sum;
}

/* Normalize vector and compute Rayleigh quotient */
__global__ void normalize_rq_kernel(const float* __restrict__ w,
                                     float* __restrict__ v,
                                     float* __restrict__ rq_scratch,
                                     float* __restrict__ norm_scratch,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Single-thread reduction for norm and RQ */
    if (idx == 0) {
        float norm_sq = 0.0f;
        float rq = 0.0f;
        for (int i = 0; i < n; i++) {
            norm_sq += w[i] * w[i];
            rq += v[i] * w[i];
        }
        float norm = sqrtf(norm_sq);
        *norm_scratch = norm;
        *rq_scratch = rq;

        if (norm > 1e-30f) {
            for (int i = 0; i < n; i++) {
                v[i] = w[i] / norm;
            }
        }
    }
}

/* Deflation: R = R - lambda * v * v^T
 * Each thread handles one element of R. */
__global__ void deflate_kernel(float* __restrict__ R,
                                const float* __restrict__ v,
                                float lambda,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;

    int row = idx / n;
    int col = idx % n;

    R[idx] -= lambda * v[row] * v[col];
}

/* Build shifted matrix M = shift*I - L on device */
__global__ void build_shifted_kernel(const float* __restrict__ L,
                                      float* __restrict__ M,
                                      float shift,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;

    int row = idx / n;
    int col = idx % n;

    float val = -L[idx];
    if (row == col) val += shift;
    M[idx] = val;
}

/* Find max diagonal element of L (for shift) */
__global__ void max_diag_kernel(const float* __restrict__ L,
                                 float* __restrict__ result,
                                 int n) {
    /* Single-thread for simplicity — for large n, use reduction */
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mx = 0.0f;
        for (int i = 0; i < n; i++) {
            float d = L[i * n + i];
            if (d > mx) mx = d;
        }
        *result = mx;
    }
}

/* Sort eigenvalues ascending with bubble sort (small n or on-host) */
/* We sort on host after copy. */

/* Copy eigenvector v into column col_idx of eigenvectors matrix (column-major) */
__global__ void store_eigenvector_kernel(const float* __restrict__ v,
                                          float* __restrict__ eigenvectors,
                                          int col_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    eigenvectors[col_idx * n + i] = v[i];
}

/* ============================================================
 * Full eigendecomposition via cuSOLVER Dn Dsyevd
 * ============================================================ */

CscCudaError csc_cuda_eigendecompose(csc_cuda_context* ctx) {
    if (!ctx) return CSC_CUDA_ERR_NULL_PTR;
    if (!ctx->laplacian_built) return CSC_CUDA_ERR_INVALID_STATE;

    int n = ctx->n;
    cusolverDnHandle_t handle = (cusolverDnHandle_t)ctx->cusolver_handle;

    /* cuSOLVER requires column-major upper/lower triangular.
     * Our Laplacian is row-major but symmetric, so row-major == column-major.
     * We use upper fill mode. */

    /* Query workspace size */
    int lwork = 0;
    cusolverStatus_t cs = cusolverDnSsyevd_bufferSize(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,  /* compute eigenvectors */
        CUBLAS_FILL_MODE_UPPER,
        n,
        ctx->d_laplacian,  /* will be overwritten */
        n,
        ctx->d_eigenvalues,
        &lwork);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cusolverDnSsyevd_bufferSize failed: %d\n", cs);
        return CSC_CUDA_ERR_CUSOLVER;
    }

    /* Allocate workspace if needed */
    if (lwork > 0) {
        if (ctx->d_solver_workspace == NULL ||
            (size_t)lwork * sizeof(float) > ctx->solver_workspace_bytes) {
            if (ctx->d_solver_workspace) cudaFree(ctx->d_solver_workspace);
            cudaError_t err = cudaMalloc(&ctx->d_solver_workspace,
                                          (size_t)lwork * sizeof(float));
            if (err != cudaSuccess) return CSC_CUDA_ERR_ALLOC;
            ctx->solver_workspace_bytes = (size_t)lwork * sizeof(float);
        }
    }

    /* Allocate info */
    int* d_info = NULL;
    cudaError_t err = cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) return CSC_CUDA_ERR_ALLOC;

    /* Copy Laplacian to eigenvectors buffer (cuSOLVER overwrites input) */
    size_t mat_bytes = (size_t)n * n * sizeof(float);
    cudaMemcpy(ctx->d_eigenvectors, ctx->d_laplacian, mat_bytes,
               cudaMemcpyDeviceToDevice);

    /* Run eigendecomposition */
    cs = cusolverDnSsyevd(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n,
        ctx->d_eigenvectors,  /* input/output: eigenvectors in columns */
        n,
        ctx->d_eigenvalues,
        (float*)ctx->d_solver_workspace,
        lwork,
        d_info);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cusolverDnSsyevd failed: %d\n", cs);
        cudaFree(d_info);
        return CSC_CUDA_ERR_CUSOLVER;
    }

    /* Check info */
    int h_info = 0;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_info);

    if (h_info != 0) {
        fprintf(stderr, "cusolverDnSsyevd: info = %d (convergence issue)\n", h_info);
        if (h_info > 0) return CSC_CUDA_ERR_CONVERGE;
        return CSC_CUDA_ERR_INTERNAL;
    }

    ctx->k = n;
    ctx->eigen_computed = true;
    return CSC_CUDA_OK;
}

/* ============================================================
 * Partial eigendecomposition via power iteration + deflation
 * ============================================================ */

CscCudaError csc_cuda_eigendecompose_partial(csc_cuda_context* ctx, int k) {
    if (!ctx) return CSC_CUDA_ERR_NULL_PTR;
    if (!ctx->laplacian_built) return CSC_CUDA_ERR_INVALID_STATE;
    if (k <= 0 || k > ctx->n) return CSC_CUDA_ERR_DIMENSION;

    int n = ctx->n;
    int block = 256;
    int grid_vec = (n + block - 1) / block;
    int grid_mat = (n * n + block - 1) / block;

    size_t mat_bytes = (size_t)n * n * sizeof(float);
    size_t vec_bytes = (size_t)n * sizeof(float);

    /* Allocate device work arrays */
    float h_shift = 0.0f;
    float* h_eigenvalues = NULL;
    float* h_v = NULL;
    float* h_evecs = NULL;
    float *d_M = NULL, *d_R = NULL, *d_v = NULL, *d_w = NULL;
    float *d_rq = NULL, *d_norm = NULL, *d_shift = NULL;

    cudaError_t err;
    err = cudaMalloc(&d_M, mat_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_R, mat_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_v, vec_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_w, vec_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_rq, sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_norm, sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_shift, sizeof(float));
    if (err != cudaSuccess) goto cleanup;

    /* Find shift = max diagonal of L */
    max_diag_kernel<<<1, 1>>>(ctx->d_laplacian, d_shift, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_shift, d_shift, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_shift < 1e-12f) h_shift = 1.0f;  /* safety */

    /* Build M = shift*I - L */
    build_shifted_kernel<<<grid_mat, block>>>(ctx->d_laplacian, d_M, h_shift, n);
    cudaDeviceSynchronize();

    /* R = M (residual matrix for deflation) */
    cudaMemcpy(d_R, d_M, mat_bytes, cudaMemcpyDeviceToDevice);

    /* Host arrays for eigenvalue tracking */
    h_eigenvalues = (float*)calloc(k, sizeof(float));
    h_v = (float*)malloc(vec_bytes);
    if (!h_eigenvalues || !h_v) goto cleanup;

    for (int ev = 0; ev < k; ev++) {
        /* Initialize v with varied seed */
        for (int i = 0; i < n; i++) {
            h_v[i] = 1.0f / (float)(i + 1 + ev * 7);
        }
        cudaMemcpy(d_v, h_v, vec_bytes, cudaMemcpyHostToDevice);

        /* Power iteration */
        float prev_lambda = 0.0f;
        int max_iter = 2000;
        float tol = 1e-10f;

        for (int iter = 0; iter < max_iter; iter++) {
            /* w = R * v */
            power_iteration_kernel<<<grid_vec, block>>>(d_R, d_v, d_w, n);
            cudaDeviceSynchronize();

            /* Normalize v = w / ||w|| and compute Rayleigh quotient */
            normalize_rq_kernel<<<1, 1>>>(d_w, d_v, d_rq, d_norm, n);
            cudaDeviceSynchronize();

            float lambda = 0.0f;
            cudaMemcpy(&lambda, d_rq, sizeof(float), cudaMemcpyDeviceToHost);

            /* Check convergence */
            if (fabsf(lambda - prev_lambda) < tol && iter > 10) {
                prev_lambda = lambda;
                break;
            }
            prev_lambda = lambda;
        }

        /* eigenvalue of L = shift - lambda_M */
        h_eigenvalues[ev] = h_shift - prev_lambda;

        /* Store eigenvector */
        store_eigenvector_kernel<<<grid_vec, block>>>(d_v, ctx->d_eigenvectors, ev, n);
        cudaDeviceSynchronize();

        /* Deflate: R = R - lambda * v * v^T */
        deflate_kernel<<<grid_mat, block>>>(d_R, d_v, prev_lambda, n);
        cudaDeviceSynchronize();
    }

    /* Copy eigenvalues to device */
    cudaMemcpy(ctx->d_eigenvalues, h_eigenvalues, k * sizeof(float),
               cudaMemcpyHostToDevice);

    /* Sort eigenvalues ascending on host, reorder eigenvectors */
    /* For simplicity, do a selection sort */
    h_evecs = (float*)malloc(mat_bytes);
    if (h_evecs) {
        cudaMemcpy(h_evecs, ctx->d_eigenvectors, mat_bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < k - 1; i++) {
            int min_idx = i;
            for (int j = i + 1; j < k; j++) {
                if (h_eigenvalues[j] < h_eigenvalues[min_idx])
                    min_idx = j;
            }
            if (min_idx != i) {
                /* Swap eigenvalues */
                float tmp = h_eigenvalues[i];
                h_eigenvalues[i] = h_eigenvalues[min_idx];
                h_eigenvalues[min_idx] = tmp;

                /* Swap eigenvector columns */
                for (int r = 0; r < n; r++) {
                    float tv = h_evecs[i * n + r];
                    h_evecs[i * n + r] = h_evecs[min_idx * n + r];
                    h_evecs[min_idx * n + r] = tv;
                }
            }
        }

        /* Copy sorted back to device */
        cudaMemcpy(ctx->d_eigenvalues, h_eigenvalues, k * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_eigenvectors, h_evecs, mat_bytes, cudaMemcpyHostToDevice);
        free(h_evecs);
    }

    free(h_eigenvalues);
    free(h_v);
    if (h_evecs) free(h_evecs);

    ctx->k = k;
    ctx->eigen_computed = true;

cleanup:
    if (d_M) cudaFree(d_M);
    if (d_R) cudaFree(d_R);
    if (d_v) cudaFree(d_v);
    if (d_w) cudaFree(d_w);
    if (d_rq) cudaFree(d_rq);
    if (d_norm) cudaFree(d_norm);
    if (d_shift) cudaFree(d_shift);

    return CSC_CUDA_OK;
}

/* ============================================================
 * Getters
 * ============================================================ */

CscCudaError csc_cuda_get_eigenvalues(csc_cuda_context* ctx, float* h_out) {
    if (!ctx || !h_out) return CSC_CUDA_ERR_NULL_PTR;
    if (!ctx->eigen_computed) return CSC_CUDA_ERR_INVALID_STATE;

    cudaMemcpy(h_out, ctx->d_eigenvalues, ctx->k * sizeof(float),
               cudaMemcpyDeviceToHost);
    return CSC_CUDA_OK;
}

CscCudaError csc_cuda_get_eigenvectors(csc_cuda_context* ctx, float* h_out) {
    if (!ctx || !h_out) return CSC_CUDA_ERR_NULL_PTR;
    if (!ctx->eigen_computed) return CSC_CUDA_ERR_INVALID_STATE;

    size_t mat_bytes = (size_t)ctx->n * ctx->k * sizeof(float);
    cudaMemcpy(h_out, ctx->d_eigenvectors, mat_bytes, cudaMemcpyDeviceToHost);
    return CSC_CUDA_OK;
}
