# ==============================================================================
# BT Benchmark Makefile (Adapted for separated headers and utils)
# ==============================================================================

PLAT   ?= cpu
OUTDIR := bin

BENCHMARK  = sp
BENCHMARKU = SP
.DEFAULT_GOAL := all

# 1. 引入基础配置
include ../../libs/utils/make.def

# 2. 目录路径定义
HEADERS = ../../libs/headers
COMMON  = ../../libs/utils

# 新增：显式指定头文件搜索路径
INCLUDES = $(C_INC) -I$(HEADERS)

# 3. 数据集映�?
_DATASET := $(strip $(DATASET))
ifdef _DATASET
    ifeq ($(_DATASET),MINI)
        override CLASS := S
    else ifeq ($(_DATASET),SMALL)
        override CLASS := W
    else ifeq ($(_DATASET),MEDIUM)
        override CLASS := A
    else ifeq ($(_DATASET),LARGE)
        override CLASS := B
    else ifeq ($(_DATASET),EXTRALARGE)
        override CLASS := C
    else
        $(error Unknown DATASET '$(_DATASET)'. Choose from: MINI SMALL MEDIUM LARGE EXTRALARGE)
    endif
endif

ifndef CLASS
$(error CLASS is not set. Use CLASS=<class> or DATASET=<dataset>)
endif

# 导出�?setparams 工具
export CLASS
export BENCHMARK

# 4. 编译器及平台参数设定 (注入 INCLUDES)
CC_cpu  := $(CC)
FL_cpu  := $(INCLUDES) $(CFLAGS) $(CLINKFLAGS)

CC_cuda := nvcc
FL_cuda := -O2 -arch=sm_80 -std=c++17 $(INCLUDES)

CC_rocm := hipcc
FL_rocm := -O2 --offload-arch=gfx90a -std=c++17 $(INCLUDES)

CC_musa := musacc
FL_musa := -O2 -std=c++17 $(INCLUDES)

# 动态选择
CXX := $(if $(CC_$(PLAT)),$(CC_$(PLAT)),$(CC))
FL  := $(if $(FL_$(PLAT)),$(FL_$(PLAT)),$(CFLAGS))

VALID := cpu cuda rocm musa
ifeq ($(filter $(PLAT),$(VALID)),)
$(error Unknown platform '$(PLAT)'. Choose from: $(VALID))
endif

# 5. 公共对象文件
COMMON_OBJS = \
    $(COMMON)/c_print_results.o \
    $(COMMON)/c_timers.o \
    $(COMMON)/c_wtime.o

# 6. setparams 工具规则 (核心修复：直接用 gcc 编译 C 源文件，不再依赖 utils 下的 Makefile)
$(COMMON)/setparams: $(COMMON)/setparams.cpp
	gcc -O2 -o $@ $<

npbparams.hpp: $(COMMON)/setparams
	$(COMMON)/setparams $(BENCHMARK) $(CLASS)

# 7. 公共对象编译规则
$(COMMON)/c_print_results.o: $(COMMON)/c_print_results.cpp
	cd $(COMMON) && $(CC) -c $(C_INC) -I../headers $(CFLAGS) c_print_results.cpp

$(COMMON)/c_timers.o: $(COMMON)/c_timers.cpp
	cd $(COMMON) && $(CC) -c $(C_INC) -I../headers $(CFLAGS) c_timers.cpp

$(COMMON)/c_wtime.o: $(COMMON)/$(WTIME)
	cd $(COMMON) && $(CC) -c $(C_INC) -I../headers $(CFLAGS) -o c_wtime.o $(WTIME)

# 8. 最终编译规�?
.PHONY: all
all: npbparams.hpp $(COMMON_OBJS)
	@mkdir -p $(OUTDIR)
	$(CXX) $(FL) c/sp.cpp $(COMMON_OBJS) $(C_LIB) -o $(OUTDIR)/$(BENCHMARK)_$(CLASS)_$(PLAT)

# 9. 清理规则
.PHONY: clean
clean:
	- rm -f $(OUTDIR)/*
	- rm -f *.o *~
	- rm -f npbparams.hpp core