#!/bin/bash
set -e

BASE_URL="https://www.jetson-ai-lab.com/code-samples/openpi_on_thor"
TARGET_DIR="openpi_on_thor"

echo "Downloading OpenPi Jetson Thor deployment scripts..."

mkdir -p "${TARGET_DIR}/patches"

FILES=(
    thor.Dockerfile
    pyproject.toml
    pi05_inference.py
    pytorch_to_onnx.py
    build_engine.sh
    trt_model_forward.py
    trt_torch.py
    calibration_data.py
)

for f in "${FILES[@]}"; do
    echo "  Downloading ${f}..."
    wget -q "${BASE_URL}/${f}" -O "${TARGET_DIR}/${f}"
done

echo "  Downloading patches/apply_gemma_fixes.py..."
wget -q "${BASE_URL}/patches/apply_gemma_fixes.py" -O "${TARGET_DIR}/patches/apply_gemma_fixes.py"

chmod +x "${TARGET_DIR}/build_engine.sh"

echo ""
echo "Done! Scripts downloaded to ${TARGET_DIR}/"
echo ""
ls "${TARGET_DIR}/"
