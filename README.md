# Conservation Spectral CUDA

GPU-accelerated spectral analysis of conservation graphs using CUDA.

Built on [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/) for eigendecomposition, [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) for sparse operations, and custom CUDA kernels for Laplacian construction, conservation analysis, and real-time tracking.

## Requirements

- **CUDA Toolkit** 11.0+ (nvcc, cuSOLVER, cuSPARSE, cuBLAS)
- **GCC** 9+ or compatible C compiler
- **CUDA-capable GPU** (Compute Capability 6.0+)

## Build

```bash
make              # Build library
make tests        # Build and run tests
make bench        # Build and run benchmarks
make all          # Build everything
```

## Project Structure

```
conservation-spectral-cuda/
├── Makefile
├── include/
│   └── conservation_spectral_cuda.h    # Public API header
├── src/
│   ├── laplacian.cu                    # GPU Laplacian construction
│   ├── eigen.cu                        # cuSOLVER eigendecomposition
│   ├── conservation.cu                 # Conservation analysis kernels
│   └── tracker.cu                      # Real-time tracker with GPU updates
├── tests/
│   ├── test_basic.cu                   # Core functionality tests
│   └── test_chords.cu                  # Musical chord graph tests
├── benchmarks/
│   └── bench_scale.cu                  # Scale benchmark (100→10000 nodes)
└── README.md
```

## Quick Start

```c
#include "conservation_spectral_cuda.h"

int main() {
    int n = 100;  // 100 vertices

    // Create GPU context
    csc_cuda_context* ctx = csc_cuda_create(n);

    // Build Laplacian from transition matrix
    float transitions[100 * 100] = { /* your data */ };
    csc_cuda_build_laplacian(ctx, transitions, false);

    // Full eigendecomposition (small-medium graphs)
    csc_cuda_eigendecompose(ctx);

    // Or partial: top-k eigenvalues via power iteration
    // csc_cuda_eigendecompose_partial(ctx, 10);

    // Spectral gap
    float gap = csc_cuda_spectral_gap(ctx);

    // Conservation ratios
    float attributes[100] = { /* vertex attributes */ };
    float ratios[100];
    csc_cuda_conservation_ratio(ctx, attributes, ratios);

    // Fingerprint
    char* fp = csc_cuda_fingerprint_compute(ctx);
    printf("Fingerprint: %s\n", fp);
    free(fp);

    // Cleanup
    csc_cuda_destroy(ctx);
    return 0;
}
```

## API Reference

### Context Lifecycle

| Function | Description |
|---|---|
| `csc_cuda_create(n)` | Create GPU context for n-vertex graph |
| `csc_cuda_destroy(ctx)` | Free all GPU memory and handles |

### Laplacian Construction

| Function | Description |
|---|---|
| `csc_cuda_build_laplacian(ctx, h_transitions, normalized)` | Build from dense host matrix |
| `csc_cuda_build_laplacian_sparse(ctx, rows, cols, weights, nnz, normalized)` | Build from sparse COO |

### Eigendecomposition

| Function | Description |
|---|---|
| `csc_cuda_eigendecompose(ctx)` | Full eigen via cuSOLVER Dsyevd |
| `csc_cuda_eigendecompose_partial(ctx, k)` | Top-k via power iteration + deflation |
| `csc_cuda_get_eigenvalues(ctx, h_out)` | Copy eigenvalues to host |
| `csc_cuda_get_eigenvectors(ctx, h_out)` | Copy eigenvectors to host |

### Conservation Analysis

| Function | Description |
|---|---|
| `csc_cuda_conservation_ratio(ctx, h_attrs, h_ratios)` | Compute all conservation ratios |
| `csc_cuda_spectral_gap(ctx)` | Largest gap between eigenvalues |
| `csc_cuda_cheeger_constant(ctx)` | Graph cut approximation |

### Real-time Tracker

| Function | Description |
|---|---|
| `csc_cuda_tracker_init(ctx, window_size)` | Initialize sliding window tracker |
| `csc_cuda_tracker_feed(ctx, observation)` | Feed value, returns 0/1/2 (nominal/warning/critical) |
| `csc_cuda_tracker_check(ctx)` | Check current state |

### Fingerprinting

| Function | Description |
|---|---|
| `csc_cuda_fingerprint_compute(ctx)` | Hex fingerprint (caller frees) |
| `csc_cuda_fingerprint_compare(fp1, fp2)` | Similarity in [0,1] |

## GPU Kernels

### `build_laplacian_kernel`
Parallel Laplacian construction: `L = D - W` (unnormalized) or `L = I - D^{-1/2} W D^{-1/2}` (normalized). Each thread computes one matrix element.

### `power_iteration_kernel`
GPU-parallel power iteration with deflation for partial eigendecomposition. Computes Rayleigh quotient and normalizes on device.

### `conservation_ratio_kernel`
Projects vertex attributes onto eigenvectors, computes gradient variance — all on GPU.

### `tracker_update_kernel`
Parallel sliding window shift and z-score anomaly detection.

## Benchmarks

Expected performance on modern GPUs:

| Graph Size | GPU Laplacian | GPU Eigen | vs CPU Speedup |
|---|---|---|---|
| 100 nodes | <1 ms | ~5 ms | 2-5x |
| 1,000 nodes | ~2 ms | ~50 ms | 10-30x |
| 10,000 nodes | ~50 ms | ~2s | 50-100x |

Run benchmarks:
```bash
make bench
# Or with custom max size:
./bench_scale --max-n 5000
```

## Error Codes

| Code | Meaning |
|---|---|
| `CSC_CUDA_OK` | Success |
| `CSC_CUDA_ERR_NO_GPU` | No CUDA GPU detected |
| `CSC_CUDA_ERR_ALLOC` | GPU memory allocation failed |
| `CSC_CUDA_ERR_CUSOLVER` | cuSOLVER internal error |
| `CSC_CUDA_ERR_CONVERGE` | Eigendecomposition did not converge |

## Related

- [conservation-spectral-c](https://github.com/SuperInstance/conservation-spectral-c) — CPU C implementation
- [conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python) — Python SDK

## License

MIT
