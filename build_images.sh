#!/usr/bin/env bash
# Build the FastPLMs base image and every per-family image (skipping Boltz for
# now; deferred until its native deps are worked out).
#
# Usage: ./build_images.sh [family ...]
#   ./build_images.sh                   # build base + all families
#   ./build_images.sh esm_plusplus      # build base + just esm_plusplus
#   ./build_images.sh --no-base esm2    # skip base rebuild, just esm2
set -euo pipefail

FAMILIES_DEFAULT=(esm2 esm_plusplus e1 dplm dplm2 ankh)

build_base=1
families=()
for arg in "$@"; do
    case "$arg" in
        --no-base) build_base=0 ;;
        *) families+=("$arg") ;;
    esac
done

if [ ${#families[@]} -eq 0 ]; then
    families=("${FAMILIES_DEFAULT[@]}")
fi

if [ $build_base -eq 1 ]; then
    echo "==> Building fastplms-base"
    docker build -f Dockerfile.base -t fastplms-base .
fi

for family in "${families[@]}"; do
    dockerfile="Dockerfile.${family}"
    if [ ! -f "$dockerfile" ]; then
        echo "Skipping ${family}: ${dockerfile} does not exist"
        continue
    fi
    echo "==> Building fastplms-${family}"
    docker build -f "$dockerfile" -t "fastplms-${family}" .
done
