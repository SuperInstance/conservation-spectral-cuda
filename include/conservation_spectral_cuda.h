/*
 * conservation_spectral_cuda.h — Conservation Spectral SDK for CUDA
 *
 * GPU-accelerated spectral analysis of conservation graphs.
 * Requires CUDA Toolkit 11.0+ and cuSOLVER/cuSPARSE.
 *
 * Usage:
 *   #include "conservation_spectral_cuda.h"
 *   Link with: -lcusolver -lcusparse -lcublas
 *
 * Version: 0.1.0
 */

#ifndef CONSERVATION_SPECTRAL_CUDA_H
#define CONSERVATION_SPECTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Version
 * ============================================================ */

#define CSC_CUDA_VERSION_MAJOR 0
#define CSC_CUDA_VERSION_MINOR 1
#define CSC_CUDA_VERSION_PATCH 0

/* ============================================================
 * Error codes
 * ============================================================ */

typedef enum {
    CSC_CUDA_OK              = 0,
    CSC_CUDA_ERR_NULL_PTR    = 1,
    CSC_CUDA_ERR_ALLOC       = 2,
    CSC_CUDA_ERR_LAUNCH      = 3,
    CSC_CUDA_ERR_CUSOLVER    = 4,
    CSC_CUDA_ERR_CUSPARSE    = 5,
    CSC_CUDA_ERR_CUBLAS      = 6,
    CSC_CUDA_ERR_DIMENSION   = 7,
    CSC_CUDA_ERR_CONVERGE    = 8,
    CSC_CUDA_ERR_NO_GPU      = 9,
    CSC_CUDA_ERR_INTERNAL    = 10,
    CSC_CUDA_ERR_INVALID_STATE = 11,
} CscCudaError;

/* Return human-readable error string */
const char* csc_cuda_strerror(CscCudaError err);

/* ============================================================
 * GPU Context — holds all GPU allocations for one graph
 * ============================================================ */

typedef struct {
    /* Laplacian (device, row-major, n×n) */
    float*  d_laplacian;        /* device memory */
    float*  d_workspace;        /* scratch for matvec etc. */

    /* Eigendecomposition results (device) */
    float*  d_eigenvalues;      /* device, length n */
    float*  d_eigenvectors;     /* device, n×n column-major */

    /* cuSOLVER handles */
    void*   cusolver_handle;    /* cusolverDnHandle_t */
    void*   cusolver_info;      /* solver info */
    void*   d_solver_workspace; /* cuSOLVER workspace */
    size_t  solver_workspace_bytes;

    /* cuSPARSE handles */
    void*   cusparse_handle;    /* cusparseHandle_t */

    /* Graph dimensions */
    int     n;                  /* number of vertices */
    int     k;                  /* number of eigenvalues computed */

    /* Tracker state */
    float*  d_tracker_history;  /* device, window_size floats */
    float   baseline_mean;
    float   baseline_std;
    int     tracker_count;
    int     tracker_window_size;
    bool    tracker_baseline_set;

    /* Flags */
    bool    laplacian_built;
    bool    eigen_computed;
} csc_cuda_context;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/* Create context for an n-vertex graph. Allocates GPU memory. */
csc_cuda_context* csc_cuda_create(int n);

/* Destroy context and free all GPU memory. */
void csc_cuda_destroy(csc_cuda_context* ctx);

/* ============================================================
 * Laplacian Construction (GPU)
 * ============================================================ */

/* Build Laplacian from a host transition/adjacency matrix.
 * h_transitions: host pointer, n×n row-major float matrix.
 * normalized:    if true, build symmetric normalized Laplacian.
 * Returns CSC_CUDA_OK on success. */
CscCudaError csc_cuda_build_laplacian(csc_cuda_context* ctx,
                                       const float* h_transitions,
                                       bool normalized);

/* Build Laplacian from a sparse edge list on host.
 * h_rows, h_cols: edge endpoints (COO format), nnz edges.
 * h_weights: edge weights, length nnz.
 * Returns CSC_CUDA_OK on success. */
CscCudaError csc_cuda_build_laplacian_sparse(csc_cuda_context* ctx,
                                              const int* h_rows,
                                              const int* h_cols,
                                              const float* h_weights,
                                              int nnz,
                                              bool normalized);

/* ============================================================
 * Eigendecomposition (GPU — cuSOLVER)
 * ============================================================ */

/* Full eigendecomposition via cuSOLVER Dn Dsyevd.
 * Computes all n eigenvalues/eigenvectors.
 * Must call csc_cuda_build_laplacian first.
 * Returns CSC_CUDA_OK on success. */
CscCudaError csc_cuda_eigendecompose(csc_cuda_context* ctx);

/* Partial eigendecomposition: compute only k smallest eigenvalues.
 * Uses power iteration + deflation on GPU for speed.
 * Returns CSC_CUDA_OK on success. */
CscCudaError csc_cuda_eigendecompose_partial(csc_cuda_context* ctx, int k);

/* ============================================================
 * Conservation Analysis (GPU)
 * ============================================================ */

/* Compute conservation ratios for all eigenvectors against an attribute array.
 * h_attributes: host pointer, n floats.
 * h_ratios:     host pointer, output n floats (caller allocated).
 * Returns CSC_CUDA_OK on success. */
CscCudaError csc_cuda_conservation_ratio(csc_cuda_context* ctx,
                                          const float* h_attributes,
                                          float* h_ratios);

/* Get the spectral gap (largest gap between consecutive eigenvalues).
 * Must call csc_cuda_eigendecompose first.
 * Returns gap value, or -1.0f on error. */
float csc_cuda_spectral_gap(csc_cuda_context* ctx);

/* Compute Cheeger constant approximation from Fiedler vector.
 * Returns the constant, or -1.0f on error. */
float csc_cuda_cheeger_constant(csc_cuda_context* ctx);

/* Copy eigenvalues to host. h_out must have space for n floats. */
CscCudaError csc_cuda_get_eigenvalues(csc_cuda_context* ctx, float* h_out);

/* Copy eigenvectors to host (column-major, n×n). h_out must have space for n*n floats. */
CscCudaError csc_cuda_get_eigenvectors(csc_cuda_context* ctx, float* h_out);

/* ============================================================
 * Real-time Tracker (GPU-accelerated sliding window)
 * ============================================================ */

/* Initialize tracker state within context. */
CscCudaError csc_cuda_tracker_init(csc_cuda_context* ctx, int window_size);

/* Feed observation, returns 0=nominal, 1=warning, 2=critical.
 * GPU-accelerated: sliding window update and z-score on device. */
int csc_cuda_tracker_feed(csc_cuda_context* ctx, float observation);

/* Check current tracker state without feeding. */
int csc_cuda_tracker_check(const csc_cuda_context* ctx);

/* ============================================================
 * Spectral Fingerprint (GPU)
 * ============================================================ */

/* Compute hex fingerprint string from eigenvalues on GPU.
 * Caller must free() the returned string. */
char* csc_cuda_fingerprint_compute(csc_cuda_context* ctx);

/* Compare two fingerprints. Returns similarity in [0,1]. */
float csc_cuda_fingerprint_compare(const char* fp1, const char* fp2);

/* ============================================================
 * Utility
 * ============================================================ */

/* Query GPU info. Writes name/VRAM into provided buffers.
 * Returns CSC_CUDA_OK if a CUDA GPU is available. */
CscCudaError csc_cuda_device_info(char* name, int name_len,
                                   size_t* total_mem_mb);

/* Print a simple benchmark header (GPU name, memory, etc.) to stdout. */
void csc_cuda_print_device_info(void);

#ifdef __cplusplus
}
#endif

#endif /* CONSERVATION_SPECTRAL_CUDA_H */
