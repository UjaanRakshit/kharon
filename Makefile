NVCC  ?= nvcc
ARCH  ?= sm_89
BUILD ?= build
INC   := -Isrc/core -Isrc/kernels

ifeq ($(OS),Windows_NT)
  CSTD := -Xcompiler /std:c11
else
  CSTD := -Xcompiler -std=c11
endif

ifdef DEBUG
  OPT := -G -g -O0 -DKHARON_DEBUG
else
  OPT := -O2 -lineinfo
endif

NVCCFLAGS := -arch=$(ARCH) $(OPT) $(INC)
LDLIBS    := -lcublas
HDRS := $(wildcard src/core/*.h src/kernels/*.h)

# On Windows, MSYS make mangles PATH for the native nvcc child, so cl.exe is not
# found via PATH. Point nvcc at it directly with -ccbin (set by scripts/build.ps1).
ifdef CCBIN
  NVCCFLAGS += -ccbin "$(CCBIN)"
endif

CORE_C  := arena refio gpt adamw ckpt data
CORE_CU := kernels
K_CU    := flash
OBJ := $(addprefix $(BUILD)/,$(addsuffix .o,$(CORE_C) $(CORE_CU) $(K_CU)))

TESTS := test_loadref test_forward test_backward test_step test_resume test_flash test_bf16fwd test_bf16step test_tp1 test_pp1
BINS  := $(addprefix $(BUILD)/,$(addsuffix .exe,$(TESTS)))
BENCH := bench_step bench_fused bench_flash bench_bf16
BENCHBINS := $(addprefix $(BUILD)/,$(addsuffix .exe,$(BENCH)))

.PHONY: all tests bench train clean
all: tests bench train
tests: $(BINS)
bench: $(BENCHBINS)
train: $(BUILD)/train.exe

$(BUILD)/train.exe: src/train.c $(OBJ) $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) $< $(OBJ) -o $@ $(LDLIBS)

# Multi-GPU (NCCL + MPI) targets — cluster only. mpicxx supplies MPI; NCCL via $NCCL_ROOT.
MPICXX  ?= mpicxx
NCCL_INC := $(if $(NCCL_ROOT),-I$(NCCL_ROOT)/include,)
NCCL_LIB := $(if $(NCCL_ROOT),-L$(NCCL_ROOT)/lib,) -lnccl
.PHONY: tp
tp: $(BUILD)/bench_allreduce.exe $(BUILD)/tp_train.exe
$(BUILD)/comms.o: src/parallel/comms.cu $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Isrc/parallel $(NCCL_INC) -ccbin $(MPICXX) -c $< -o $@
$(BUILD)/bench_allreduce.exe: bench/bench_allreduce.c $(BUILD)/comms.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) -Isrc/parallel $(NCCL_INC) -ccbin $(MPICXX) $< $(BUILD)/comms.o -o $@ $(NCCL_LIB)
$(BUILD)/tp_train.exe: src/tp_train.c $(BUILD)/comms.o $(OBJ) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) -Isrc/parallel $(NCCL_INC) -ccbin $(MPICXX) $< $(BUILD)/comms.o $(OBJ) -o $@ $(LDLIBS) $(NCCL_LIB)

# Coarse but portable: any header change rebuilds all objects (no nvcc/MSVC -MMD).
$(BUILD)/%.o: src/core/%.c $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) -c $< -o $@

$(BUILD)/%.o: src/core/%.cu $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/%.o: src/kernels/%.cu $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/%.exe: tests/%.c $(OBJ) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) $< $(OBJ) -o $@ $(LDLIBS)

$(BUILD)/%.exe: bench/%.c $(OBJ) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) $< $(OBJ) -o $@ $(LDLIBS)

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
