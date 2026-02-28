NVCC ?= nvcc

ifndef GPU
GPU = H100
endif

ifndef PYTHON_VERSION
PYTHON_VERSION = 3.12
endif

BASE_NVCCFLAGS = -DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda \
            --expt-relaxed-constexpr -Xcompiler=-Wno-psabi \
            -Xcompiler=-fno-strict-aliasing --use_fast_math \
            -forward-unknown-to-host-compiler -O3 -std=c++20 \
            -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt \
            -lcudart_static -lcublas -lineinfo

BASE_NVCCFLAGS += $(shell python3 -m pybind11 --includes) \
             $(shell python3-config --ldflags) \
             -shared -fPIC -lpython$(PYTHON_VERSION)

# Optional call-site supplied flags (e.g., extra torch include paths).
BASE_NVCCFLAGS += $(EXTRA_NVCCFLAGS)

ifeq ($(GPU),H100)
BASE_NVCCFLAGS += -DKITTENS_HOPPER -arch=sm_90a
else ifeq ($(GPU),B200)
BASE_NVCCFLAGS += -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else ifeq ($(GPU),A100)
BASE_NVCCFLAGS += -DKITTENS_A100 -arch=sm_80
else
$(error Unsupported GPU: $(GPU). Use H100, B200, or A100)
endif

# Preserve base flags even if callers override NVCCFLAGS from CLI.
ALL_NVCCFLAGS = $(BASE_NVCCFLAGS) $(NVCCFLAGS)

EXT_SUFFIX = $(shell python3-config --extension-suffix)
TARGETS = tp_sp$(EXT_SUFFIX) fmha_attention$(EXT_SUFFIX) zigzag_attention$(EXT_SUFFIX)

all: $(TARGETS)

tp_sp$(EXT_SUFFIX): tp_sp.cu tp_sp.cuh multimem.cuh distributed.cuh qwen3.cuh
	$(NVCC) tp_sp.cu $(ALL_NVCCFLAGS) -o $@

fmha_attention$(EXT_SUFFIX): fmha_attention.cu fmha_attention.cuh qwen3.cuh
	$(NVCC) fmha_attention.cu $(ALL_NVCCFLAGS) -o $@

zigzag_attention$(EXT_SUFFIX): zigzag_attention.cu zigzag_attention.cuh qwen3.cuh
	$(NVCC) zigzag_attention.cu $(ALL_NVCCFLAGS) -lnccl -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
