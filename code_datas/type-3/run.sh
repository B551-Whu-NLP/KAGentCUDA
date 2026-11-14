#!/bin/bash
# ==============================================================================
# PolyBench Run Script
# ------------------------------------------------------------------------------
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   -b, --bench    NAME     Benchmark name (e.g., gemm, 2mm, covariance)
#   -p, --plat     PLAT     Platform (cpu|cuda|rocm|musa), default: cpu
#   -v, --verbose           Enable verbose output
#   -h, --help              Show this help
#
# Examples:
#   ./run.sh -b gemm -p cuda
#   ./run.sh -b fdtd-2d -p cpu --verbose
# ==============================================================================

set -e

PLAT="cpu"
BENCH=""
VERBOSE=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--bench)    BENCH=$2;     shift 2 ;;
        -p|--plat)     PLAT=$2;      shift 2 ;;
        -v|--verbose)  VERBOSE=1;    shift ;;
        -h|--help)     head -17 "$0" | grep "^#" | sed 's/^# \?//'; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z $BENCH ]]; then
    echo "[ERROR] Must specify -b BENCHMARK (e.g., gemm, 2mm)"
    echo "        Available benchmarks can be seen via 'ls *.c'"
    exit 1
fi

# ── Check Binary & Auto-Build ─────────────────────────────────────────────────
BIN="./bin/${BENCH}_${PLAT}"

if [[ ! -f "$BIN" ]]; then
    echo "[INFO] Binary not found: $BIN"
    echo "[INFO] Triggering auto-build: make $BENCH PLAT=$PLAT"
    echo "-----------------------------------------------------------"
    make "$BENCH" PLAT="$PLAT"
    echo "-----------------------------------------------------------"
    
    if [[ ! -f "$BIN" ]]; then
        echo "[ERROR] Auto-build failed or binary still missing: $BIN"
        exit 1
    fi
fi

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ $VERBOSE -eq 1 ]]; then
    echo "==========================================================="
    echo "[INFO] Benchmark : $BENCH"
    echo "[INFO] Platform  : $PLAT"
    echo "[INFO] Executing : $BIN"
    echo "==========================================================="
fi

# 执行二进制程序
if [[ $VERBOSE -eq 1 ]]; then
    "$BIN" --verbose
else
    "$BIN"
fi