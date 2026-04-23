#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-pat3d-sam3}"
SAM3_REPO="${SAM3_REPO:-${HOME}/.cache/pat3d/sam3}"
SAM3_CHECKPOINT_DIR="${SAM3_CHECKPOINT_DIR:-${HOME}/.cache/pat3d/sam3-checkpoints}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0.dev20260421+cu130}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.27.0.dev20260421+cu130}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0.dev20260421+cu130}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu130}"

mkdir -p "$(dirname "${SAM3_REPO}")"
if [[ ! -d "${SAM3_REPO}" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/sam3 "${SAM3_REPO}"
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12 pip
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade "pip<27" "setuptools<81" wheel
conda run -n "${ENV_NAME}" python -m pip install --pre \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --extra-index-url "${PYTORCH_INDEX_URL}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade -e "${SAM3_REPO}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade huggingface_hub decord pycocotools einops psutil

mkdir -p "${SAM3_CHECKPOINT_DIR}"
conda run -n "${ENV_NAME}" python - <<'PY' "${SAM3_CHECKPOINT_DIR}"
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

output_dir = Path(sys.argv[1]).expanduser()
output_dir.mkdir(parents=True, exist_ok=True)
for filename in ("config.json", "sam3.pt"):
    cached = Path(hf_hub_download(repo_id="facebook/sam3", filename=filename))
    target = output_dir / filename
    shutil.copy2(cached, target)
    print(filename, "->", target)
PY

echo "SAM 3 environment ready. Python executable:"
conda run -n "${ENV_NAME}" python -c "import sys; print(sys.executable)"
conda run -n "${ENV_NAME}" python -c "import torch, torchvision, torchaudio, sam3; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; build_sam3_image_model(checkpoint_path='${SAM3_CHECKPOINT_DIR}/sam3.pt', load_from_HF=False, device='cpu'); print(torch.__version__, torchvision.__version__, torchaudio.__version__)"
