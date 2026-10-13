# ==============================================================================
# ParaShift 增强型通用 Makefile
# ==============================================================================

TARGET          ?= _unset_
PLAT            ?= cpu
OUTDIR          := bin

# 校验平台
VALID_PLATS := cpu omp cuda rocm musa
ifeq ($(filter $(PLAT),$(VALID_PLATS)),)
$(error Unknown platform '$(PLAT)'. Choose from: $(VALID_PLATS))
endif

# ── 基础编译参数 ──────────────────────────────────────────────────────────────
# 增加 -I. 确保能找到同目录下的自定义头文件 (如计时器)
INCLUDES := -I. -I../../libs/headers

ifeq ($(PLAT),cpu)
    CXX     := gcc
    #CFLAGS  := -O3 -Wall $(INCLUDES)
    #LDFLAGS := -lm
    CFLAGS  := -O3 -Wall -fopenmp $(INCLUDES)
    LDFLAGS := -fopenmp -lm

else ifeq ($(PLAT),omp)
    CXX     := gcc
    # 增加 -fopenmp，同时保留对 C++ 源文件的兼容
    CFLAGS  := -O3 -Wall -fopenmp $(INCLUDES)
    LDFLAGS := -fopenmp -lm

else ifeq ($(PLAT),cuda)
    CXX     := nvcc
    # 增加 -Xcompiler -fopenmp 以支持 CUDA 代码中混合使用 OpenMP
    CFLAGS  := -O2 -arch=sm_80 $(INCLUDES) -Xcompiler -fopenmp
    LDFLAGS := -lm -lgomp

else ifeq ($(PLAT),rocm)
    CXX     := hipcc
    CFLAGS  := -O2 -fopenmp $(INCLUDES)
    LDFLAGS := -lm -lgomp

else ifeq ($(PLAT),musa)
    CXX     := /usr/local/musa/bin/mcc
    CFLAGS  := -O2 -fopenmp $(INCLUDES) \
               -I/usr/local/musa/include
    LDFLAGS := -L/usr/local/musa/lib -lmusart -lm -lgomp
endif

# ── 源码推断逻辑 ──────────────────────────────────────────────────────────────
# 自动探测源文件，支持 .c, .cpp, .cu, .hip, .mu
ifeq ($(PLAT),cpu)
    SRC := $(firstword $(wildcard $(TARGET).cpp $(TARGET).c))
    CXX := $(if $(filter %.cpp,$(SRC)),g++,gcc)
else ifeq ($(PLAT),omp)
    SRC := $(firstword $(wildcard $(TARGET).cpp $(TARGET).c))
    CXX := $(if $(filter %.cpp,$(SRC)),g++,gcc)
else ifeq ($(PLAT),cuda)
    SRC := $(TARGET).cu
else ifeq ($(PLAT),rocm)
    SRC := $(TARGET).hip
else ifeq ($(PLAT),musa)
    SRC := $(TARGET).mu
endif

BIN := $(OUTDIR)/$(TARGET)_$(PLAT)

# ── 构建规则 ──────────────────────────────────────────────────────────────────
.PHONY: all clean info check_target build
.SUFFIXES:

all: build

%:
	@$(MAKE) TARGET=$@ PLAT=$(PLAT) build

check_target:
	@if [ "$(TARGET)" = "_unset_" ]; then \
		echo "[ERROR] 请指定 TARGET=<name>"; exit 1; \
	fi
	@if [ -z "$(SRC)" ]; then \
		echo "[ERROR] 找不到目标 '$(TARGET)' 对应的源文件"; exit 1; \
	fi

build: check_target
	@mkdir -p $(OUTDIR)
	$(CXX) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

clean:
	rm -rf $(OUTDIR)

info:
	@echo "SRC     : $(SRC)"
	@echo "BIN     : $(BIN)"
	@echo "LDFLAGS : $(LDFLAGS)"
