#!/bin/zsh
set -euo pipefail

cd "${0:A:h:h}"
mise install python
uv venv --python "$(mise which python)" .venv
uv pip install --python .venv/bin/python numpy numba mlx-lm "mlx-audio[stt]" huggingface_hub hf_xet
HF_HUB_DISABLE_XET=1 .venv/bin/python Scripts/download_models.py
zsh Scripts/build.sh
print "安裝完成。打開 Build/LiveTranslator.app 即可開始。"
