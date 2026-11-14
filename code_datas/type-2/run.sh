#!/bin/bash
# ==============================================================================
# Universal Benchmark Run Script
# ------------------------------------------------------------------------------
# Usage:
#   ./run.sh TARGET [OPTIONS]
#
# Options:
#   -d, --dataset  NAME     Dataset name (SMALL | MEDIUM | LARGE)
#   -p, --plat     PLAT     Platform (cpu|omp|cuda|rocm|musa), default: cpu
#   -t, --threads  N        Number of threads, default: 4
#   -v, --verbose           Save output to file
#   -h, --help              Show this help
#
# Examples:
#   ./run.sh hotspot   -d LARGE -p cpu
#   ./run.sh nn        -d SMALL -p omp -t 8
#   ./run.sh heartwall -d LARGE -p cpu --verbose
# ==============================================================================

set -e

PLAT="cpu"
DATASET_INPUT=""
THREADS=4
VERBOSE=0

# ==============================================================================
# ★ 数据路径配置：新增有数据依赖的 benchmark 时在此处添加一行
# ==============================================================================
HOTSPOT_DATA_DIR="../../libs/input_datas/hotspot"
HEARTWALL_DATA_DIR="../../libs/input_datas/heartwall"
NN_DATA_DIR="../../libs/input_datas/nn"
NEEDLE_DATA_DIR="../../libs/input_datas/needle"
# CFD_DATA_DIR="../../libs/input_datas/cfd"
# ==============================================================================

# ── TARGET (第一个位置参数) ───────────────────────────────────────────────────
if [[ $# -lt 1 || $1 == -* ]]; then
    echo "[ERROR] Must specify TARGET as first argument (e.g. hotspot, c_347)"
    exit 1
fi
TARGET=$1; shift

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)  DATASET_INPUT=$2; shift 2 ;;
        -p|--plat)     PLAT=$2;          shift 2 ;;
        -t|--threads)  THREADS=$2;       shift 2 ;;
        -v|--verbose)  VERBOSE=1;        shift ;;
        -h|--help)     head -22 "$0" | grep "^#" | sed 's/^# \?//'; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z $DATASET_INPUT ]]; then
    echo "[ERROR] Must specify -d DATASET (SMALL, MEDIUM, or LARGE)"
    exit 1
fi
DATASET=${DATASET_INPUT^^}

# 输出文件统一命名：<TARGET>_<DATASET>_<PLAT>_out.txt
OUT_FILE="${TARGET}_${DATASET}_${PLAT}_out.txt"

# ── Per-target: 数据路径 & 参数构造 ──────────────────────────────────────────
case $TARGET in

    hotspot)
        case $DATASET in
            SMALL)  GRID=64   ;;
            MEDIUM) GRID=512  ;;
            LARGE)  GRID=1024 ;;
            *) echo "[ERROR] Unknown DATASET '$DATASET'"; exit 1 ;;
        esac
        SIM_TIME=2
        TEMP_FILE="${HOTSPOT_DATA_DIR}/temp_${GRID}"
        POWER_FILE="${HOTSPOT_DATA_DIR}/power_${GRID}"
        [[ ! -f "$TEMP_FILE"  ]] && { echo "[ERROR] Not found: $TEMP_FILE";  exit 1; }
        [[ ! -f "$POWER_FILE" ]] && { echo "[ERROR] Not found: $POWER_FILE"; exit 1; }
        [[ $VERBOSE -eq 1 ]] && OUT_ARG="$OUT_FILE" || OUT_ARG="/dev/null"
        RUN_ARGS="$GRID $GRID $SIM_TIME $THREADS $TEMP_FILE $POWER_FILE $OUT_ARG"
        ;;

    nn)
        # 参数：<filelist> <k> <lat> <lng>
        # filelist 里存相对路径，需 cd 到数据目录再执行
        case $DATASET in
            SMALL)  FILELIST="filelist.txt" ;;
            MEDIUM) FILELIST="filelist.txt" ;;
            LARGE)  FILELIST="filelist.txt" ;;
            *) echo "[ERROR] Unknown DATASET '$DATASET'"; exit 1 ;;
        esac
        K=5
        LAT=30
        LNG=90
        ABS_BIN="$(realpath "./bin/nn_${PLAT}")"
        ABS_DATA="$(realpath "$NN_DATA_DIR")"
        [[ ! -f "${ABS_DATA}/$FILELIST" ]] && { echo "[ERROR] Not found: ${ABS_DATA}/$FILELIST"; exit 1; }
        export RESULT_DIR="$(pwd)"
        echo "[INFO] Running: $ABS_BIN $FILELIST $K $LAT $LNG  (in ${ABS_DATA})"
        cd "$ABS_DATA"
        [[ $VERBOSE -eq 1 ]] && VERBOSE_FLAG="--verbose" || VERBOSE_FLAG=""
        exec "$ABS_BIN" "$FILELIST" "$K" "$LAT" "$LNG" $VERBOSE_FLAG
        ;;

    needle)
        case $DATASET in
            SMALL)  GSIZE=1024 ;;
            MEDIUM) GSIZE=2048 ;;
            LARGE)  GSIZE=4096 ;;
            *) echo "[ERROR] Unknown DATASET '$DATASET'"; exit 1 ;;
        esac
        PENALTY=10
        # 必须加上 $VERBOSE_FLAG
        [[ $VERBOSE -eq 1 ]] && VERBOSE_FLAG="--verbose" || VERBOSE_FLAG=""
        RUN_ARGS="$GSIZE $PENALTY $THREADS $VERBOSE_FLAG"
        ;;

    # ── 新增有数据依赖的 benchmark 在此添加 case ────────────────────────────
    # newbench)
    #     ...
    #     RUN_ARGS="..."
    #     ;;
    srad)
        # 参数映射：rows cols y1 y2 x1 x2 nthreads lambda niter
        case $DATASET in
            SMALL)
                ROWS=512; COLS=512; Y1=0; Y2=127; X1=0; X2=127; NITER=10 ;;
            MEDIUM)
                ROWS=1024; COLS=1024; Y1=0; Y2=255; X1=0; X2=255; NITER=20 ;;
            LARGE)
                ROWS=2048; COLS=2048; Y1=0; Y2=511; X1=0; X2=511; NITER=50 ;;
            *) echo "[ERROR] Unknown DATASET '$DATASET'"; exit 1 ;;
        esac
        LAMBDA=0.5
        # 注意：srad 通常接受线程数作为第7个参数
        RUN_ARGS="$ROWS $COLS $Y1 $Y2 $X1 $X2 $THREADS $LAMBDA $NITER"
        
        # 处理 verbose 逻辑（如果你的 srad 版本支持输出重定向或 --verbose）
        [[ $VERBOSE -eq 1 ]] && RUN_ARGS="$RUN_ARGS --verbose"
        ;;

    *)
        # 无数据依赖的 benchmark（c_347、c_21 等）：无额外参数
        [[ $VERBOSE -eq 1 ]] && VERBOSE_FLAG="--verbose" || VERBOSE_FLAG=""
        RUN_ARGS="$VERBOSE_FLAG"
        ;;

esac

# ── Check Binary ──────────────────────────────────────────────────────────────
BIN="./bin/${TARGET}_${PLAT}"
[[ ! -f "$BIN" ]] && { echo "[ERROR] Binary not found: $BIN (run: make TARGET=$TARGET PLAT=$PLAT)"; exit 1; }

# ── Run ───────────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=$THREADS

echo "[INFO] Running: $BIN $RUN_ARGS"
$BIN $RUN_ARGS