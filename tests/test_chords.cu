/*
 * test_chords.cu — Musical chord graph tests for Conservation Spectral CUDA
 *
 * Models musical chords as graphs where:
 *   - Vertices = notes (with frequency attribute)
 *   - Edges = consonance relationships between notes
 *
 * Tests conservation analysis on chord structures:
 *   - Major triad, minor triad, diminished, augmented, 7th chords
 *   - Verifies that consonant chords have distinct spectral signatures
 *   - Compares fingerprints across chord types
 *
 * Run: ./test_chords
 */

#include "conservation_spectral_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %-45s ", #name); fflush(stdout); } while (0)
#define PASS() do { tests_passed++; printf("[PASS]\n"); } while (0)
#define FAIL(msg) printf("[FAIL] %s\n", msg)
#define ASSERT(cond, msg) if (!(cond)) { FAIL(msg); return; }
#define ASSERT_APPROX(a, b, tol, msg) if (fabsf((a)-(b)) >= (tol)) { FAIL(msg); return; }

/* Note frequencies (C4 = 261.63 Hz) */
#define NOTE_C4  261.63f
#define NOTE_D4  293.66f
#define NOTE_E4  329.63f
#define NOTE_F4  349.23f
#define NOTE_G4  392.00f
#define NOTE_A4  440.00f
#define NOTE_B4  493.88f
#define NOTE_Bb4 466.16f

/* Consonance weight between two frequencies.
 * Higher = more consonant. Based on simple ratio approximation. */
static float consonance(float f1, float f2) {
    float ratio = f1 / f2;
    if (ratio > 1.0f) ratio = 1.0f / ratio;

    /* Perfect intervals get higher weights */
    if (fabsf(ratio - 1.0f) < 0.01f) return 1.0f;     /* unison */
    if (fabsf(ratio - 0.5f) < 0.02f) return 0.95f;     /* octave */
    if (fabsf(ratio - 0.667f) < 0.02f) return 0.9f;    /* perfect fifth */
    if (fabsf(ratio - 0.75f) < 0.02f) return 0.85f;    /* perfect fourth */
    if (fabsf(ratio - 0.6f) < 0.02f) return 0.8f;      /* major third */
    if (fabsf(ratio - 0.625f) < 0.02f) return 0.7f;    /* minor third */
    if (fabsf(ratio - 0.5625f) < 0.02f) return 0.4f;   /* tritone */

    /* Default: dissonance based on distance from simple ratios */
    return 0.2f;
}

/* Build a chord graph from frequency array.
 * Returns a filled transition matrix (adjacency with consonance weights). */
static void build_chord_matrix(float* W, const float* freqs, int n) {
    memset(W, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                W[i * n + j] = consonance(freqs[i], freqs[j]);
            }
        }
    }
}

/* ============================================================
 * Tests
 * ============================================================ */

static void test_major_triad(void) {
    TEST(major_triad_spectral);

    int n = 3;
    float freqs[] = { NOTE_C4, NOTE_E4, NOTE_G4 }; /* C major */
    float W[9];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    CscCudaError err = csc_cuda_eigendecompose(ctx);
    ASSERT(err == CSC_CUDA_OK, "eigen failed");

    float evals[3];
    csc_cuda_get_eigenvalues(ctx, evals);

    /* First eigenvalue should be ~0 (connected graph) */
    ASSERT_APPROX(evals[0], 0.0f, 0.5f, "eigenvalue[0] not ~0");

    /* Spectral gap should be positive */
    float gap = csc_cuda_spectral_gap(ctx);
    ASSERT(gap > 0.0f, "spectral gap <= 0");

    printf("(gap=%.3f) ", gap);

    csc_cuda_destroy(ctx);
    PASS();
}

static void test_minor_triad(void) {
    TEST(minor_triad_spectral);

    int n = 3;
    float freqs[] = { NOTE_A4, NOTE_C4, NOTE_E4 }; /* A minor */
    float W[9];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float gap = csc_cuda_spectral_gap(ctx);
    ASSERT(gap > 0.0f, "spectral gap <= 0");

    printf("(gap=%.3f) ", gap);

    csc_cuda_destroy(ctx);
    PASS();
}

static void test_diminished_chord(void) {
    TEST(diminished_chord);

    int n = 3;
    /* Diminished: C-Eb-Gb — more dissonant */
    float freqs[] = { NOTE_C4, NOTE_D4 * 1.05946f * 1.05946f, NOTE_F4 * 0.94387f };
    float W[9];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float evals[3];
    csc_cuda_get_eigenvalues(ctx, evals);
    float gap = csc_cuda_spectral_gap(ctx);

    ASSERT(gap >= 0.0f, "gap negative");

    printf("(gap=%.3f) ", gap);
    csc_cuda_destroy(ctx);
    PASS();
}

static void test_seventh_chord(void) {
    TEST(seventh_chord_4note);

    int n = 4;
    /* G7: G-B-D-F */
    float freqs[] = { NOTE_G4, NOTE_B4, NOTE_D4, NOTE_F4 };
    float W[16];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float evals[4];
    csc_cuda_get_eigenvalues(ctx, evals);

    /* Connected graph: first eigenvalue ~0 */
    ASSERT_APPROX(evals[0], 0.0f, 0.5f, "not connected");

    float gap = csc_cuda_spectral_gap(ctx);
    ASSERT(gap > 0.0f, "gap <= 0");

    printf("(gap=%.3f) ", gap);
    csc_cuda_destroy(ctx);
    PASS();
}

static void test_conservation_major_vs_minor(void) {
    TEST(conservation_major_vs_minor);

    int n = 3;

    /* Major: C-E-G */
    float freqs_major[] = { NOTE_C4, NOTE_E4, NOTE_G4 };
    float W_maj[9];
    build_chord_matrix(W_maj, freqs_major, n);

    /* Minor: A-C-E */
    float freqs_minor[] = { NOTE_A4, NOTE_C4, NOTE_E4 };
    float W_min[9];
    build_chord_matrix(W_min, freqs_minor, n);

    /* Major */
    csc_cuda_context* ctx_maj = csc_cuda_create(n);
    ASSERT(ctx_maj, "major create failed");
    csc_cuda_build_laplacian(ctx_maj, W_maj, false);
    csc_cuda_eigendecompose(ctx_maj);
    char* fp_maj = csc_cuda_fingerprint_compute(ctx_maj);
    ASSERT(fp_maj, "major fingerprint NULL");

    /* Minor */
    csc_cuda_context* ctx_min = csc_cuda_create(n);
    ASSERT(ctx_min, "minor create failed");
    csc_cuda_build_laplacian(ctx_min, W_min, false);
    csc_cuda_eigendecompose(ctx_min);
    char* fp_min = csc_cuda_fingerprint_compute(ctx_min);
    ASSERT(fp_min, "minor fingerprint NULL");

    /* Fingerprints should differ (different chords) */
    float sim = csc_cuda_fingerprint_compare(fp_maj, fp_min);
    printf("(similarity=%.3f) ", sim);

    /* Different chords should have some difference */
    ASSERT(sim < 1.0f, "different chords identical fingerprint");

    free(fp_maj);
    free(fp_min);
    csc_cuda_destroy(ctx_maj);
    csc_cuda_destroy(ctx_min);
    PASS();
}

static void test_chord_tracker_monitoring(void) {
    TEST(chord_tracker_monitoring);

    int n = 3;
    float freqs[] = { NOTE_C4, NOTE_E4, NOTE_G4 };
    float W[9];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    /* Use tracker to monitor spectral gap over time */
    csc_cuda_tracker_init(ctx, 8);

    /* Feed normal spectral gaps */
    float baseline_gap = csc_cuda_spectral_gap(ctx);
    for (int i = 0; i < 8; i++) {
        csc_cuda_tracker_feed(ctx, baseline_gap);
    }

    /* Check: feeding same value should be nominal */
    int alert = csc_cuda_tracker_feed(ctx, baseline_gap);
    ASSERT(alert == 0, "false alert on stable gap");

    /* Feed a dramatically different value (simulating chord change) */
    alert = csc_cuda_tracker_feed(ctx, baseline_gap + 100.0f);
    ASSERT(alert >= 1, "missed chord change alert");

    printf("(baseline_gap=%.3f) ", baseline_gap);
    csc_cuda_destroy(ctx);
    PASS();
}

static void test_augmented_chord(void) {
    TEST(augmented_chord);

    int n = 3;
    /* Augmented: C-E-G# — symmetrical intervals */
    float freqs[] = { NOTE_C4, NOTE_E4, NOTE_G4 * 1.05946f };
    float W[9];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float evals[3];
    csc_cuda_get_eigenvalues(ctx, evals);

    /* Augmented triad is highly symmetrical — eigenvalues should reflect that */
    float gap = csc_cuda_spectral_gap(ctx);
    ASSERT(gap >= 0.0f, "gap negative");

    printf("(evals=[%.2f,%.2f,%.2f], gap=%.3f) ", evals[0], evals[1], evals[2], gap);

    csc_cuda_destroy(ctx);
    PASS();
}

static void test_power_chord(void) {
    TEST(power_chord_2note);

    int n = 2;
    /* Power chord: C-G (root + fifth) */
    float freqs[] = { NOTE_C4, NOTE_G4 };
    float W[4];
    build_chord_matrix(W, freqs, n);

    csc_cuda_context* ctx = csc_cuda_create(n);
    ASSERT(ctx, "create failed");

    csc_cuda_build_laplacian(ctx, W, false);
    csc_cuda_eigendecompose(ctx);

    float evals[2];
    csc_cuda_get_eigenvalues(ctx, evals);

    /* 2-node graph: eigenvalues 0 and sum of weights*2 */
    ASSERT_APPROX(evals[0], 0.0f, 0.5f, "not zero eigenvalue");
    ASSERT(evals[1] > 0.0f, "second eigenvalue not positive");

    printf("(evals=[%.2f,%.2f]) ", evals[0], evals[1]);

    csc_cuda_destroy(ctx);
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Conservation Spectral CUDA — Chord Graph Tests ===\n\n");

    csc_cuda_print_device_info();
    printf("\n");

    test_major_triad();
    test_minor_triad();
    test_diminished_chord();
    test_seventh_chord();
    test_conservation_major_vs_minor();
    test_chord_tracker_monitoring();
    test_augmented_chord();
    test_power_chord();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
