#!/usr/bin/env bash
# Usage:
#   bash run.sh <stem> <platform> [--verbose]
#
# Examples:
#   bash run.sh c_1 cpu
#   bash run.sh c_1 cuda --verbose

set -euo pipefail

STEM="${1:-}"
PLAT="${2:-}"
VERBOSE=0
[ "${3:-}" = "--verbose" ] && VERBOSE=1

if [ -z "$STEM" ] || [ -z "$PLAT" ]; then
    echo "Usage: bash run.sh <stem> <platform> [--verbose]"
    exit 1
fi

case "$PLAT" in
    cpu|cuda|rocm|musa) ;;
    *) echo "[ERROR] Unknown platform '$PLAT'. Choose from: cpu cuda rocm musa"; exit 1 ;;
esac

BIN="bin/${STEM}_${PLAT}"
if [ ! -f "$BIN" ]; then
    echo "[ERROR] Binary not found: $BIN"
    echo "        Run: make $STEM PLAT=$PLAT"
    exit 1
fi

if [ $VERBOSE -eq 1 ]; then
    "$BIN" --verbose
else
    "$BIN"
fi