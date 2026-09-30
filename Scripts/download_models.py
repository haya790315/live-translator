import os
import platform
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


root = Path(__file__).resolve().parent.parent / "Models"
root.mkdir(exist_ok=True)

models = [("mlx-community/Hy-MT2-1.8B-8bit", "Hy-MT2-1.8B-8bit")]
macos = int(platform.mac_ver()[0].split(".")[0] or 0)
if macos < 26 or os.environ.get("LIVE_TRANSLATOR_PARAKEET") == "1":
    models += [
        ("mlx-community/parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt_ctc-0.6b-ja"),
        ("mlx-community/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3"),
    ]

for model, name in models:
    snapshot_download(repo_id=model, local_dir=root / name, ignore_patterns=["mlx_model/*"])

(root / "silero-vad").mkdir(exist_ok=True)
vad = hf_hub_download("onnx-community/silero-vad", "onnx/model.onnx")
(root / "silero-vad" / "model.onnx").write_bytes(Path(vad).read_bytes())
