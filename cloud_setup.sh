#!/bin/bash
# Hunyuan3D Studio - One-Command Cloud Setup
# Usage: curl -s <paste.rs URL> | bash
# Run this in Vast.ai Jupyter Terminal after renting an instance
# (PyTorch template, RTX 3090, 80GB+ disk)

set -e

echo "=============================================="
echo "  Hunyuan3D Studio - Auto Setup"
echo "=============================================="

# Step 1: Clone & install
echo ""
echo "[1/7] Cloning Hunyuan3D-2..."
cd /workspace
if [ ! -d "Hunyuan3D-2" ]; then
    git clone https://github.com/Tencent/Hunyuan3D-2.git
fi
cd Hunyuan3D-2

echo ""
echo "[2/7] Installing Python dependencies..."
pip install -r requirements.txt
pip install -e .
pip install flask trimesh pymeshlab pygltflib xatlas tiktoken sentencepiece protobuf

echo ""
echo "[3/7] Installing system dependencies..."
apt-get update && apt-get install -y libopengl0

# Step 2: Fix CUDA headers
echo ""
echo "[4/7] Setting up CUDA headers..."
CUSPARSE_PATH=$(find / -name "cusparse.h" 2>/dev/null | head -1)
if [ -n "$CUSPARSE_PATH" ]; then
    CUDA_INCLUDE_DIR=$(dirname "$CUSPARSE_PATH")
    export CPLUS_INCLUDE_PATH=$CUDA_INCLUDE_DIR:$CPLUS_INCLUDE_PATH
    export C_INCLUDE_PATH=$CUDA_INCLUDE_DIR:$C_INCLUDE_PATH
    export CUDA_HOME=/usr/local/cuda
    echo "    Found CUDA headers at: $CUDA_INCLUDE_DIR"
else
    echo "    WARNING: cusparse.h not found. Texture build may fail."
fi

# Step 3: Build texture components
echo ""
echo "[5/7] Building texture components..."
cd hy3dgen/texgen/custom_rasterizer && python setup.py install && cd ../../..
cd hy3dgen/texgen/differentiable_renderer && python setup.py install && cd ../../..

# Step 4: Patch texture pipeline for speed (Unity game mode)
echo ""
echo "[6/8] Patching texture pipeline for speed..."
# Light removal: 50 steps -> 25 steps (good balance of speed/quality)
sed -i 's/num_inference_steps=50/num_inference_steps=25/' hy3dgen/texgen/utils/dehighlight_utils.py
# Multiview diffusion: 30 steps -> 20 steps (good balance of speed/quality)
sed -i 's/num_inference_steps=30/num_inference_steps=20/' hy3dgen/texgen/utils/multiview_utils.py
echo "    Patched: delight 50->25 steps, multiview 30->20 steps"

# Step 5: Create directories & download app files
echo ""
echo "[7/8] Downloading app files..."
mkdir -p outputs uploads static jobs
echo "    Downloading from GitHub..."
curl -sL https://raw.githubusercontent.com/SarpTolga/hunyuanMadness/main/cloud_app.py -o app.py
curl -sL https://raw.githubusercontent.com/SarpTolga/hunyuanMadness/main/static/index.html -o static/index.html
echo "    app.py: $(wc -c < app.py) bytes"
echo "    index.html: $(wc -c < static/index.html) bytes"

# Step 5: Fix Caddy proxy
echo ""
echo "[8/8] Configuring Caddy proxy..."
sed -i 's/localhost:18384/localhost:3333/g' /etc/Caddyfile && caddy reload --config /etc/Caddyfile

echo ""
echo "=============================================="
echo "  Setup complete! Starting server..."
echo "=============================================="
echo ""

# Start the server
cd /workspace/Hunyuan3D-2 && python app.py
