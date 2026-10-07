#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

ENV_NAME="Geometric"
PYTHON_VER="3.11"

echo "================================================================="
echo " Creating Conda Environment: ${ENV_NAME} (Python ${PYTHON_VER})"
echo "================================================================="

# 1. Create and activate Conda environment
conda create -n "${ENV_NAME}" python="${PYTHON_VER}" numpy scipy matplotlib ipython pip -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "================================================================="
echo " Installing PyTorch with CUDA 12.1"
echo "================================================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "================================================================="
echo " Installing PyTorch Geometric & Pre-compiled C++ Extensions"
echo "================================================================="
pip install torch_geometric

TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")

echo "Detected PyTorch Version: ${TORCH_VER} | CUDA Tag: ${CUDA_VER}"

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html"

echo "================================================================="
echo " Installing Scientific, Optimization & Domain Packages"
echo "================================================================="
pip install scikit-learn scikit-optimize networkx cvxpy h5py
pip install obspy scikit-fmm
pip install pyyaml ruamel.yaml

echo "================================================================="
echo " Verification Test"
echo "================================================================="
python -c "
import torch
import torch_cluster
import torch_geometric
print(f'[✓] PyTorch Version : {torch.__version__}')
print(f'[✓] CUDA Available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[✓] GPU Model       : {torch.cuda.get_device_name(0)}')
print(f'[✓] torch_cluster   : {torch_cluster.__version__}')
"

echo "================================================================="
echo " Setup complete! Activate with: conda activate ${ENV_NAME}"
echo "================================================================="
