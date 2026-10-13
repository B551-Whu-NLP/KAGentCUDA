# ==============================================================================
# C baseline Makefile (-lm 置于末尾以满足链接顺序要求)
# ==============================================================================

PLAT   ?= cpu
OUTDIR := bin

# ── 平台配置 ──────────────────────────────────────────────────────────────────
# CPU (C11)
CC_cpu   := gcc
EXT_cpu  := c
FL_cpu   := -O2 -std=c11 -Wall
LD_cpu   := -lm

# CUDA (C++17)
CC_cuda  := nvcc
EXT_cuda := cu
FL_cuda  := -O2 -arch=sm_80 -std=c++17
LD_cuda  := 

# ROCm (C++17)
CC_rocm  := hipcc
EXT_rocm := cpp
FL_rocm  := -O2 --offload-arch=gfx90a -std=c++17
LD_rocm  := -lm

# MUSA (C++17)
CC_musa  := musacc
EXT_musa := mu
FL_musa  := -O2 -std=c++17
LD_musa  := -lm

# ── 变量解析 ──────────────────────────────────────────────────────────────────
CC  := $(CC_$(PLAT))
EXT := $(EXT_$(PLAT))
FL  := $(FL_$(PLAT))
LDFLAGS := $(LD_$(PLAT))

VALID := cpu cuda rocm musa
ifeq ($(filter $(PLAT),$(VALID)),)
$(error Unknown platform '$(PLAT)'. Choose from: $(VALID))
endif

# ── 构建规则 ──────────────────────────────────────────────────────────────────
# 特别说明：我们将 $(LDFLAGS) 放在了命令的最后
%:
	$(eval SRC := $(firstword \
		$(wildcard src/$@.$(EXT)) \
		$(wildcard generated/$(PLAT)/$@.$(EXT)) \
		$(wildcard $@.$(EXT)) \
	))
	@[ -n "$(SRC)" ] || { \
		echo "[ERROR] Source not found for target $@ with extension .$(EXT)"; \
		exit 1; }
	@mkdir -p $(OUTDIR)
	$(CC) $(FL) $(SRC) -o $(OUTDIR)/$@_$(PLAT) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OUTDIR)/*