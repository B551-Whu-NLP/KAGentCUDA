#!/bin/bash
# ==============================================================================
# BT Benchmark Run Script
# ------------------------------------------------------------------------------
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   -c, --class   CLASS    NPB class (S|W|A|B|C|D|E)
#   -d, --dataset DATASET  PolyBench-style dataset (MINI|SMALL|MEDIUM|LARGE|EXTRALARGE)
#   -p, --plat    PLAT     Platform (cpu|cuda|rocm|musa), default: cpu
#   -v, --verbose          Enable verbose output to bt_out.txt
#   -h, --help             Show this help
#
# Examples:
#   ./run.sh -c A -p cpu
#   ./run.sh -d LARGE -p cuda -v
# ==============================================================================

set -e

PLAT=cpu
CLASS=
DATASET=
VERBOSE=

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--class)    CLASS=$2;   shift 2 ;;
        -d|--dataset)  DATASET=$2; shift 2 ;;
        -p|--plat)     PLAT=$2;    shift 2 ;;
        -v|--verbose)  VERBOSE=verbose; shift ;;
        -h|--help)     head -20 "$0" | grep "^#" | sed 's/^# \?//'; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ── DATASET → CLASS mapping ───────────────────────────────────────────────────
if [[ -n $DATASET ]]; then
    case $DATASET in
        MINI)        CLASS=S ;;
        SMALL)       CLASS=W ;;
        MEDIUM)      CLASS=A ;;
        LARGE)       CLASS=B ;;
        EXTRALARGE)  CLASS=C ;;
        *) echo "[ERROR] Unknown DATASET '$DATASET'. Choose from: MINI SMALL MEDIUM LARGE EXTRALARGE"; exit 1 ;;
    esac
fi

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z $CLASS ]]; then
    echo "[ERROR] Must specify either -c CLASS or -d DATASET"
    exit 1
fi

BIN="bin/ft_${CLASS}_${PLAT}"
if [[ ! -f $BIN ]]; then
    echo "[ERROR] Binary not found: $BIN"
    echo "        Please build first: make CLASS=$CLASS PLAT=$PLAT"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[INFO] Running: $BIN $VERBOSE"
$BIN $VERBOSE