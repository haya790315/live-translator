import os
from pathlib import Path

from huggingface_hub import snapshot_download


root = Path(__file__).resolve().parent.parent / "Models"
root.mkdir(exist_ok=True)

# 載入 PLaMo 時會執行 repo 附的 modeling_plamo_mlx.py，所以釘住已檢視過的 revision，不跟著 main 走
models = [
    ("pfnet/plamo-3-translate-2b-MLX-4bit", "plamo-3-translate-2b-MLX-4bit", "63777ddc6dc80aacacd92a1bf6966105f2f6270c"),
    # 每句定稿後用它重解該句音訊
    ("mlx-community/Qwen3-ASR-1.7B-8bit", "Qwen3-ASR-1.7B-8bit", None),
]
if os.environ.get("LIVE_TRANSLATOR_PLAMO_8B") == "1":
    # 較準、多佔約 3.7 GB 記憶體。revision 是 2026-11-06 從 HF API 查到的 sha；執行的是 repo 附的程式碼，換版前要重新檢視
    models.append(("pfnet/plamo-3-translate-8b-MLX-4bit", "plamo-3-translate-8b-MLX-4bit", "9d87684408bdf34500b7d6115c257d83350c1fe5"))

if os.environ.get("LIVE_TRANSLATOR_POLISH") == "1":
    # 翻譯前潤稿（選用）：Qwen3-4B 4bit，約 2.3 GB。revision 是 2026-11-06 從 HF API 查到的 sha。啟用在 App 裡按「潤稿」
    models.append(("mlx-community/Qwen3-4B-4bit", "Qwen3-4B-4bit", "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"))

for model, name, revision in models:
    snapshot_download(repo_id=model, revision=revision, local_dir=root / name, ignore_patterns=["mlx_model/*"])

# 預設開啟 Qwen 重解；已有設定就不動，使用者關掉的不會被打開
choice = root / "final-model.txt"
if not choice.exists():
    choice.write_text("Qwen3-ASR-1.7B-8bit\n")
