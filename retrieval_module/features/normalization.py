# ============================================================
# 归一化配置
# 12维数据操作特征，C侧和CUDA侧使用统一基准
# ============================================================

# C侧特征顺序（严格按此顺序输出向量）
C_FEATURE_ORDER = [
    'indirect_access_ratio',       # 0
    'consecutive_access_ratio',    # 1
    'offset_access_ratio',         # 2
    'has_stencil_pattern',         # 3
    'stencil_neighbor_count_norm', # 4
    'self_product_diversity',      # 5
    'fma_ratio',                   # 6
    'compute_intensity',           # 7
    'read_write_ratio',            # 8
    'strided_access_ratio',        # 9
    'spatial_locality_score',      # 10
    'reduction_count_norm',        # 11
]

# CUDA侧特征顺序（与C侧语义严格对应）
CUDA_FEATURE_ORDER = [
    'indirect_access_ratio',       # 0  ↔ indirect_access_ratio
    'coalesced_access_ratio',      # 1  ↔ consecutive_access_ratio
    'offset_access_ratio',         # 2  ↔ offset_access_ratio
    'has_stencil_pattern',         # 3  ↔ has_stencil_pattern
    'stencil_neighbor_count_norm', # 4  ↔ stencil_neighbor_count_norm
    'self_product_diversity',      # 5  ↔ self_product_diversity
    'fma_ratio',                   # 6  ↔ fma_ratio
    'compute_intensity',           # 7  ↔ compute_intensity
    'read_write_ratio',            # 8  ↔ read_write_ratio
    'strided_access_ratio',        # 9  ↔ strided_access_ratio
    'spatial_locality_score',      # 10 ↔ spatial_locality_score
    'reduction_count_norm',        # 11 ↔ reduction_count_norm
]

# 归一化函数（两侧统一基准）
NORM_FUNCS = {
    'indirect_access_ratio':       lambda x: float(x),
    'consecutive_access_ratio':    lambda x: float(x),
    'coalesced_access_ratio':      lambda x: min(float(x), 1.0),
    'offset_access_ratio':         lambda x: min(float(x), 1.0),
    'has_stencil_pattern':         lambda x: 1.0 if x else 0.0,
    'stencil_neighbor_count_norm': lambda x: float(x),
    'self_product_diversity':      lambda x: 1.0 if x else 0.0,
    'fma_ratio':                   lambda x: min(float(x), 1.0),
    'compute_intensity':           lambda x: float(x),
    'read_write_ratio':            lambda x: min(float(x), 1.0),
    'strided_access_ratio':        lambda x: min(float(x), 1.0),
    'spatial_locality_score':      lambda x: min(float(x), 1.0),
    'reduction_count_norm':        lambda x: float(x),
}

FEATURE_DIM = 12