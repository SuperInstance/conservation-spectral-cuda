# Conservation Spectral CUDA — Makefile
#
# Build the CUDA-accelerated conservation spectral library.
#
# Targets:
#   all       - Build library, tests, and benchmarks
#   lib       - Build the shared library
#   tests     - Build and run tests
#   bench     - Build and run benchmarks
#   clean     - Remove build artifacts

CUDA_PATH  ?= /usr/local/cuda
NVCC       := $(CUDA_PATH)/bin/nvcc
CC         := gcc

# Architecture flags — target sm_60+ (Pascal and up)
NVCC_FLAGS := -std=c++14 -O2 -arch=sm_60 \
              --generate-code arch=compute_60,code=[sm_60,compute_60] \
              --generate-code arch=compute_70,code=[sm_70,compute_70] \
              --generate-code arch=compute_75,code=[sm_75,compute_75] \
              --generate-code arch=compute_80,code=[sm_80,compute_80] \
              --generate-code arch=compute_86,code=[sm_86,compute_86] \
              -Xcompiler -fPIC

INCLUDES  := -Iinclude
LIBS      := -lcusolver -lcusparse -lcublas -lcudart -lm

SRCDIR    := src
TESTDIR   := tests
BENCHDIR  := benchmarks
BUILDDIR  := build

SOURCES   := $(wildcard $(SRCDIR)/*.cu)
OBJECTS   := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(SOURCES))

.PHONY: all lib tests bench clean

all: lib tests bench

# ---- Build directory ----
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ---- Compile .cu to .o ----
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# ---- Shared library ----
lib: $(BUILDDIR)/libconservation_spectral_cuda.so

$(BUILDDIR)/libconservation_spectral_cuda.so: $(OBJECTS)
	$(NVCC) -shared -o $@ $^ $(LIBS)

# ---- Tests ----
tests: $(BUILDDIR)/test_basic $(BUILDDIR)/test_chords
	@echo ""
	@echo "=== Running basic tests ==="
	./$(BUILDDIR)/test_basic
	@echo ""
	@echo "=== Running chord tests ==="
	./$(BUILDDIR)/test_chords

$(BUILDDIR)/test_basic: $(TESTDIR)/test_basic.cu $(OBJECTS) | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< $(OBJECTS) -o $@ $(LIBS)

$(BUILDDIR)/test_chords: $(TESTDIR)/test_chords.cu $(OBJECTS) | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< $(OBJECTS) -o $@ $(LIBS)

# ---- Benchmarks ----
bench: $(BUILDDIR)/bench_scale
	@echo ""
	@echo "=== Running scale benchmark ==="
	./$(BUILDDIR)/bench_scale --max-n 1000

$(BUILDDIR)/bench_scale: $(BENCHDIR)/bench_scale.cu $(OBJECTS) | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< $(OBJECTS) -o $@ $(LIBS)

# ---- Clean ----
clean:
	rm -rf $(BUILDDIR)
