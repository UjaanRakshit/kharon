NVCC  ?= nvcc
ARCH  ?= sm_89
BUILD ?= build
INC   := -Isrc/core

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

# On Windows, MSYS make mangles PATH for the native nvcc child, so cl.exe is not
# found via PATH. Point nvcc at it directly with -ccbin (set by scripts/build.ps1).
ifdef CCBIN
  NVCCFLAGS += -ccbin "$(CCBIN)"
endif

CORE_C  := arena refio gpt adamw
CORE_CU := kernels
OBJ := $(addprefix $(BUILD)/,$(addsuffix .o,$(CORE_C) $(CORE_CU)))

TESTS := test_loadref test_forward test_backward test_step
BINS  := $(addprefix $(BUILD)/,$(addsuffix .exe,$(TESTS)))

.PHONY: all tests clean
all: tests
tests: $(BINS)

$(BUILD)/%.o: src/core/%.c | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) -c $< -o $@

$(BUILD)/%.o: src/core/%.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/%.exe: tests/%.c $(OBJ) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CSTD) $< $(OBJ) -o $@ $(LDLIBS)

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
