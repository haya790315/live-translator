import os
import platform
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


root = Path(__file__).resolve().parent.parent / "Models"
root.mkdir(exist_ok=True)

models = [("mlx-community/Hy-MT2-1.8B-8bit", "Hy-MT2-1.8B-8bit")]
macos = int(platform.mac_ver()[0].split(".")[0] or 0)
# Apple 模式每句定稿後用它重解；parakeet 模式下是選用的「Qwen 重解」
models.append(("mlx-community/Qwen3-ASR-1.7B-8bit", "Qwen3-ASR-1.7B-8bit"))
if macos < 26 or os.environ.get("LIVE_TRANSLATOR_PARAKEET") == "1":
    models += [
        ("mlx-community/parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt_ctc-0.6b-ja"),
        ("mlx-community/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3"),
    ]

for model, name in models:
    snapshot_download(repo_id=model, local_dir=root / name, ignore_patterns=["mlx_model/*"])

# macOS 26 以上預設開啟 Qwen 重解（Apple 模式不翻譯，記憶體夠）；已有設定就不動，使用者關掉的不會被打開
choice = root / "final-model.txt"
if macos >= 26 and not choice.exists():
    choice.write_text("Qwen3-ASR-1.7B-8bit\n")

(root / "silero-vad").mkdir(exist_ok=True)
vad = hf_hub_download("onnx-community/silero-vad", "onnx/model.onnx")
(root / "silero-vad" / "model.onnx").write_bytes(Path(vad).read_bytes())
