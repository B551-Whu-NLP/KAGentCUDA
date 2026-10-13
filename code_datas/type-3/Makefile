# ==============================================================================
# PolyBench Benchmark Makefile (按需编译版)
# 
# Platforms: cpu (default) | cuda | rocm | musa
# Datasets : MINI_DATASET | SMALL_DATASET | MEDIUM_DATASET | LARGE_DATASET | EXTRALARGE_DATASET
#
# Usage:
#   make 2mm                    -> 编译 2mm (默认 CPU)
#   make gemm PLAT=cuda         -> 编译 gemm (使用 CUDA)
#   make fdtd-2d PLAT=musa      -> 编译 fdtd-2d (使用 MUSA)
# ==============================================================================

PLAT        ?= cpu
DATASET     ?= LARGE_DATASET
OUTDIR      := bin

# 路径配置
HEADERS_DIR := ../../libs/headers
UTILS_DIR   := ../../libs/utils
POLY_SRC    := $(UTILS_DIR)/polybench.c

# 扫描当前目录下的所有 .c 文件作为可用的 Target 列表 (例如: 2mm, gemm)
# 修改扫描规则：同时扫描 .c 和 .cu 文件
SRCS        := $(wildcard *.c) $(wildcard *.cu)
# 提取目标名（移除 .c 和 .cu 后缀）
TARGETS     := $(SRCS:.c=)
TARGETS     := $(TARGETS:.cu=)

# 修改编译规则：支持 .c 和 .cu 两种源文件
$(OUTDIR)/%_$(PLAT): %.c $(POLY_SRC)
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) $< $(POLY_SRC) $(LDFLAGS) -o $@
	@echo "✅ Compiled $< -> $@"

$(OUTDIR)/%_$(PLAT): %.cu $(POLY_SRC)
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) $< $(POLY_SRC) $(LDFLAGS) -o $@
	@echo "✅ Compiled $< -> $@"

# 基础宏定义
COMMON_FLAGS := -O3 -I$(HEADERS_DIR) -I$(UTILS_DIR) \
                -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -D$(DATASET)

# ==============================================================================
# 平台配置 (移除 OMP)
# ==============================================================================

ifeq ($(PLAT),cpu)
    CC      := gcc
    CFLAGS  := $(COMMON_FLAGS)
    LDFLAGS := -lm

else ifeq ($(PLAT),cuda)
    CC      := nvcc
    CFLAGS  := $(COMMON_FLAGS) -arch=sm_80 -x cu -fmad=false
    LDFLAGS := -lm

else ifeq ($(PLAT),rocm)
    CC      := hipcc
    CFLAGS  := $(COMMON_FLAGS) -x hip -fmad=false
    LDFLAGS := -lm

else ifeq ($(PLAT),musa)
    CC      := /usr/local/musa/bin/mcc
    CFLAGS  := $(COMMON_FLAGS) -x mu -fmad=false \
               -I/usr/local/musa-3.1.0/include \
               -I/usr/include/c++/11 \
               -I/usr/include/x86_64-linux-gnu/c++/11
    LDFLAGS := -L/usr/local/musa-3.1.0/lib \
               -L/usr/lib/gcc/x86_64-linux-gnu/11 \
               -lmusart -lstdc++ -lm
else
    $(error Unknown platform '$(PLAT)'. Choose: cpu | cuda | rocm | musa)
endif

# ==============================================================================
# 编译规则
# ==============================================================================

.PHONY: all clean info $(TARGETS)

# 如果只输入 make，不执行全量编译，而是打印提示信息
all:
	@echo "=========================================================="
	@echo "请指定要编译的 benchmark 名称！"
	@echo "用法示例: make gemm PLAT=cuda"
	@echo "可用列表: $(TARGETS)"
	@echo "=========================================================="

$(TARGETS): %: $(OUTDIR)/%_$(PLAT)

# 真正的编译动作
$(OUTDIR)/%_$(PLAT): %.c $(POLY_SRC)
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) $< $(POLY_SRC) $(LDFLAGS) -o $@
	@echo "✅ Compiled $< -> $@"

# ==============================================================================
# 清理与调试
# ==============================================================================

clean:
	rm -rf $(OUTDIR)

info:
	@echo "PLAT          : $(PLAT)"
	@echo "DATASET       : $(DATASET)"
	@echo "CC            : $(CC)"
	@echo "CFLAGS        : $(CFLAGS)"
	@echo "LDFLAGS       : $(LDFLAGS)"