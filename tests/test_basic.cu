/*
 * test_basic.cu — Basic unit tests for Conservation Spectral CUDA
 *
 * Tests: context lifecycle, Laplacian construction, eigendecomposition,
 *        conservation ratios, spectral gap, tracker.
 *
 * Run: ./test_basic
 */

#include "conservation_spectral_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST %-40s ", #name); \
    fflush(stdout); \
} while (0)

#define PASS() do { \
    tests_passed++; \
    printf("[PASS]\n"); \
} while (0)

#define FAIL(msg) do { \
    printf("[FAIL] %s\n", msg); \
} while (0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while (0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEQ(a, b, msg) ASSERT((a) != (b), msg)
#define ASSERT_APPROX(a, b, tol, msg) ASSERT(fabsf((a) - (b)) < (tol), msg)

/* ============================================================
 * Test: Context create/destroy
 * ============================================================ */
static void test_context_lifecycle(void) {
    TEST(context_lifecycle);

    csc_cuda_context* ctx = csc_cuda_create(10);
    ASSERT_NEQ(ctx, NULL, "create returned NULL");
    ASSERT_EQ(ctx->n, 10, "n mismatch");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Laplacian construction (3-node triangle)
 * ============================================================ */
static void test_laplacian_triangle(void) {
    TEST(laplacian_triangle);

    int n = 3;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    /* Triangle graph: 3 nodes, each connected to the other two, weight 1 */
    float W[9] = {
        0.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f,
    };

    CscCudaError err = csc_cuda_build_laplacian(ctx, W, false);
    ASSERT_EQ(err, CSC_CUDA_OK, "build_laplacian failed");
    ASSERT(ctx->laplacian_built, "laplacian_built not set");

    /* Read back Laplacian */
    float L[9];
    cudaMemcpy(L, ctx->d_laplacian, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    /* Expected unnormalized Laplacian for triangle:
     * D = diag(2, 2, 2), W = adjacency
     * L = D - W = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]
     */
    ASSERT_APPROX(L[0], 2.0f, 0.01f, "L[0][0] != 2");
    ASSERT_APPROX(L[1], -1.0f, 0.01f, "L[0][1] != -1");
    ASSERT_APPROX(L[4], 2.0f, 0.01f, "L[1][1] != 2");
    ASSERT_APPROX(L[8], 2.0f, 0.01f, "L[2][2] != 2");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Eigendecomposition (triangle)
 * ============================================================ */
static void test_eigen_triangle(void) {
    TEST(eigen_triangle);

    int n = 3;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    float W[9] = {
        0.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f,
    };

    csc_cuda_build_laplacian(ctx, W, false);

    CscCudaError err = csc_cuda_eigendecompose(ctx);
    ASSERT_EQ(err, CSC_CUDA_OK, "eigendecompose failed");
    ASSERT(ctx->eigen_computed, "eigen_computed not set");

    float evals[3];
    csc_cuda_get_eigenvalues(ctx, evals);

    /* Triangle Laplacian eigenvalues: 0, 3, 3 */
    /* Sorted ascending */
    ASSERT_APPROX(evals[0], 0.0f, 0.1f, "evals[0] != 0");
    ASSERT_APPROX(evals[1], 3.0f, 0.1f, "evals[1] != 3");
    ASSERT_APPROX(evals[2], 3.0f, 0.1f, "evals[2] != 3");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Partial eigendecomposition
 * ============================================================ */
static void test_eigen_partial(void) {
    TEST(eigen_partial);

    int n = 5;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    /* Build a small graph */
    float W[25] = {0};
    /* Chain: 0-1-2-3-4 */
    W[0 * 5 + 1] = W[1 * 5 + 0] = 1.0f;
    W[1 * 5 + 2] = W[2 * 5 + 1] = 1.0f;
    W[2 * 5 + 3] = W[3 * 5 + 2] = 1.0f;
    W[3 * 5 + 4] = W[4 * 5 + 3] = 1.0f;

    csc_cuda_build_laplacian(ctx, W, false);

    CscCudaError err = csc_cuda_eigendecompose_partial(ctx, 3);
    ASSERT_EQ(err, CSC_CUDA_OK, "partial eigen failed");
    ASSERT_EQ(ctx->k, 3, "k != 3");

    float evals[3];
    csc_cuda_get_eigenvalues(ctx, evals);

    /* Chain graph eigenvalues start with 0 (smallest) */
    ASSERT_APPROX(evals[0], 0.0f, 0.5f, "evals[0] != 0");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Conservation ratio
 * ============================================================ */
static void test_conservation_ratio(void) {
    TEST(conservation_ratio);

    int n = 4;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    /* Complete graph K4 */
    float W[16] = {
        0, 1, 1, 1,
        1, 0, 1, 1,
        1, 1, 0, 1,
        1, 1, 1, 0,
    };

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float attrs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float ratios[4] = {0};

    CscCudaError err = csc_cuda_conservation_ratio(ctx, attrs, ratios);
    ASSERT_EQ(err, CSC_CUDA_OK, "conservation_ratio failed");

    /* Ratios should be non-negative (variance) */
    for (int i = 0; i < n; i++) {
        ASSERT(ratios[i] >= -0.01f, "negative ratio");
    }

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Spectral gap
 * ============================================================ */
static void test_spectral_gap(void) {
    TEST(spectral_gap);

    int n = 3;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    float W[9] = {
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    };

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float gap = csc_cuda_spectral_gap(ctx);
    /* Triangle: eigenvalues 0, 3, 3. Gap = 3. */
    ASSERT_APPROX(gap, 3.0f, 0.2f, "spectral gap wrong");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Tracker
 * ============================================================ */
static void test_tracker(void) {
    TEST(tracker);

    int n = 5;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    CscCudaError err = csc_cuda_tracker_init(ctx, 10);
    ASSERT_EQ(err, CSC_CUDA_OK, "tracker init failed");

    /* Feed normal values */
    for (int i = 0; i < 10; i++) {
        int alert = csc_cuda_tracker_feed(ctx, 5.0f);
        ASSERT(alert == 0, "unexpected alert during baseline");
    }
    ASSERT(ctx->tracker_baseline_set, "baseline not set after 10 feeds");

    /* Feed more normal values */
    for (int i = 0; i < 5; i++) {
        int alert = csc_cuda_tracker_feed(ctx, 5.0f);
        ASSERT(alert == 0, "unexpected alert for normal value");
    }

    /* Feed an extreme value */
    int alert = csc_cuda_tracker_feed(ctx, 100.0f);
    /* Should be at least warning (z-score >> 3) */
    ASSERT(alert >= 1, "expected alert for extreme value");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Sparse Laplacian
 * ============================================================ */
static void test_laplacian_sparse(void) {
    TEST(laplacian_sparse);

    int n = 4;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    /* Chain: 0-1-2-3 */
    int rows[] = {0, 1, 2};
    int cols[] = {1, 2, 3};
    float weights[] = {1.0f, 1.0f, 1.0f};

    CscCudaError err = csc_cuda_build_laplacian_sparse(ctx, rows, cols,
                                                        weights, 3, false);
    ASSERT_EQ(err, CSC_CUDA_OK, "sparse laplacian failed");

    /* Read back and verify */
    float L[16];
    cudaMemcpy(L, ctx->d_laplacian, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    /* Expected: chain Laplacian
     * [[1,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,1]]
     */
    ASSERT_APPROX(L[0], 1.0f, 0.01f, "L[0][0] != 1");
    ASSERT_APPROX(L[5], 2.0f, 0.01f, "L[1][1] != 2");
    ASSERT_APPROX(L[10], 2.0f, 0.01f, "L[2][2] != 2");
    ASSERT_APPROX(L[15], 1.0f, 0.01f, "L[3][3] != 1");
    ASSERT_APPROX(L[1], -1.0f, 0.01f, "L[0][1] != -1");

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Test: Device info
 * ============================================================ */
static void test_device_info(void) {
    TEST(device_info);

    char name[256] = {0};
    size_t mem_mb = 0;
    CscCudaError err = csc_cuda_device_info(name, sizeof(name), &mem_mb);

    if (err == CSC_CUDA_OK) {
        printf("(GPU: %s, %.0f MB) ", name, (float)mem_mb);
    } else {
        printf("(No GPU) ");
    }

    PASS();
}

/* ============================================================
 * Test: Fingerprint
 * ============================================================ */
static void test_fingerprint(void) {
    TEST(fingerprint);

    int n = 3;
    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT_NEQ(ctx, NULL, "create failed");

    float W[9] = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    char* fp1 = csc_cuda_fingerprint_compute(ctx);
    ASSERT_NEQ(fp1, NULL, "fingerprint NULL");

    /* Same graph should produce same fingerprint */
    char* fp2 = csc_cuda_fingerprint_compute(ctx);
    ASSERT_NEQ(fp2, NULL, "fingerprint2 NULL");

    float sim = csc_cuda_fingerprint_compare(fp1, fp2);
    ASSERT_APPROX(sim, 1.0f, 0.01f, "identical fingerprints differ");

    free(fp1);
    free(fp2);
    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Conservation Spectral CUDA — Basic Tests ===\n\n");

    csc_cuda_print_device_info();
    printf("\n");

    test_context_lifecycle();
    test_laplacian_triangle();
    test_eigen_triangle();
    test_eigen_partial();
    test_conservation_ratio();
    test_spectral_gap();
    test_tracker();
    test_laplacian_sparse();
    test_device_info();
    test_fingerprint();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
