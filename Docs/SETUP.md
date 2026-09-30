# 開發環境建置與啟動

在一台 Apple Silicon Mac 上從零裝好、編譯並啟動 App 的步驟。只改了程式要重編，看第 6 節；只想切換模型，看第 9 節。

## 前提

| 項目 | 要求 |
|---|---|
| 硬體 | Apple Silicon Mac；記憶體 16 GB 以上（翻譯模型常駐約 2.3 GB） |
| macOS | 26 以上用 SpeechAnalyzer 辨識。14 到 25 也能執行，改用 parakeet 辨識（`Info.plist` 的 `LSMinimumSystemVersion` 是 14.0） |
| 磁碟 | 至少 5 GB（翻譯模型 1.8 GB、Python 環境 1.3 GB）；parakeet 模式再加 4.6 GB |
| 網路 | 只在安裝時需要；執行時 worker 強制離線 |

需要四個工具：

| 工具 | 用途 | 安裝 |
|---|---|---|
| Xcode Command Line Tools | `swiftc` 編譯 App、`codesign` 簽章 | `xcode-select --install` |
| Homebrew | 安裝下面兩個 | https://brew.sh |
| mise | 依 `mise.toml` 安裝 Python 3.12 | `brew install mise` |
| uv | 建立虛擬環境、安裝 Python 套件 | `brew install uv` |

確認：

```bash
xcrun swift --version
mise --version
uv --version
```

**期待輸出**（版本號可以更新）

```
Apple Swift version 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101)
2026.9.4 macos-arm64 (2026-09-09)
uv 0.12.9 (Homebrew 2026-09-01 aarch64-apple-darwin)
```

## 變數

| 變數 | 意義 | 取得元 | 例 |
|---|---|---|---|
| `REPO_ROOT` | repo 所在目錄 | 自己決定 clone 到哪 | `$HOME/work/live-translator` |

> 開新的終端機視窗時，重新執行下面這塊。

```bash
# ─── 只改這裡 ───
export REPO_ROOT="$HOME/work/live-translator"   # ← 要變更
```

## 1. 取得原始碼

```bash
git clone https://github.com/haya790315/live-translator.git "${REPO_ROOT}"
```

## 2. 一次完成安裝

第 3 到 6 節全部由一個腳本執行；想分步驟做或只重做其中一步，再看各節。

```bash
cd "${REPO_ROOT}"
zsh Scripts/setup.sh
```

**期待輸出**（最後一行）

```
安裝完成。打開 Build/LiveTranslator.app 即可開始。
```

所需時間主要在下載模型：macOS 26 約 1.8 GB，每秒 2.5 MB 的連線約 12 分鐘；parakeet 模式約 6.4 GB、45 分鐘。

## 3. 建立 Python 環境

```bash
cd "${REPO_ROOT}"
mise install python
uv venv --python "$(mise which python)" .venv
uv pip install --python .venv/bin/python numpy scikit-learn mlx-lm parakeet-mlx onnxruntime huggingface_hub hf_xet
```

確認：

```bash
"${REPO_ROOT}/.venv/bin/python" -c "import platform, mlx.core, parakeet_mlx, mlx_lm, onnxruntime; print(platform.machine(), mlx.core.__version__)"
```

**期待輸出**

```
arm64 0.32.3
```

第一個值必須是 `arm64`。經 Rosetta 執行的 x86_64 Python 裝不了 MLX。

## 4. 下載模型

模型放在 `Models/` 下，目錄名固定，App 依目錄名尋找。辨識在 macOS 26 上由系統內建的 SpeechAnalyzer 負責，不需要下載模型；腳本在 macOS 26 以上只下載翻譯模型與 VAD，14 到 25 另外下載兩個 parakeet。

| 目錄（`Models/` 下） | 用途 | 大小 | 何時下載 | 來源 | 授權 |
|---|---|---:|---|---|---|
| `Hy-MT2-1.8B-8bit` | 中文翻譯 | 1.8 GB | 一律 | mlx-community/Hy-MT2-1.8B-8bit | Apache-2.0 |
| `silero-vad/model.onnx` | 語音活動偵測，parakeet 模式用 | 2 MB | 一律 | onnx-community/silero-vad | MIT |
| `parakeet-tdt_ctc-0.6b-ja` | 日文辨識，parakeet 模式用 | 2.3 GB | macOS 25 以下 | mlx-community/parakeet-tdt_ctc-0.6b-ja | CC-BY-4.0 |
| `parakeet-tdt-0.6b-v3` | 英文辨識，parakeet 模式用 | 2.3 GB | macOS 25 以下 | mlx-community/parakeet-tdt-0.6b-v3 | CC-BY-4.0 |

在 macOS 26 上也想留 parakeet 當備援時，下載前設定 `LIVE_TRANSLATOR_PARAKEET=1`。

```bash
cd "${REPO_ROOT}"
HF_HUB_DISABLE_XET=1 .venv/bin/python Scripts/download_models.py
```

`HF_HUB_DISABLE_XET=1` 讓 Hugging Face 用一般 HTTP 下載，避免 xet 傳輸卡住。中斷後重跑同一命令會接著下載。

確認：

```bash
ls "${REPO_ROOT}"/Models/*/model.safetensors "${REPO_ROOT}/Models/silero-vad/model.onnx"
```

**期待輸出**（macOS 26；parakeet 模式另有兩個 `parakeet-*/model.safetensors`）

```
.../Models/Hy-MT2-1.8B-8bit/model.safetensors
.../Models/silero-vad/model.onnx
```

## 5. 訓練斷句分類器

parakeet 模式用一個字尾 n-gram 分類器判斷「句子講完了沒」，SpeechAnalyzer 模式不需要。權重不進 Git，安裝時在本機訓練，約 1 分鐘。

```bash
cd "${REPO_ROOT}"
.venv/bin/python Scripts/make_segment_data.py Models/bsd Models/segment-data
.venv/bin/python Scripts/train_segmenter.py Models/segment-data Models/segmenter
```

第一個命令從 GitHub 下載 BSD（Business Scene Dialogue）語料 7.8 MB 到 `Models/bsd/`，合成訓練資料到 `Models/segment-data/`；第二個訓練並輸出 `Models/segmenter/weights.json`。

**期待輸出**（第二個命令的最後兩行）

```
train rows=... test rows=... test accuracy=0.95..
saved ... weights to Models/segmenter/weights.json
```

沒有 `weights.json` 時 worker 仍能執行，只用規則斷句。

> **注意**：BSD 語料為 CC BY-NC-SA 4.0，訓練出的分類器只能非商用。

## 6. 編譯 App

```bash
cd "${REPO_ROOT}"
zsh Scripts/build.sh
```

**期待輸出**

```
已建立 Build/LiveTranslator.app（臨時簽章；執行 zsh Scripts/make_signing_identity.sh 可固定簽章，重編後不必再授權）
```

腳本先編到暫存目錄再換名，App 開著的時候重編也不會被殺掉。App 必須留在 `Build/` 裡，因為它從自己的位置往上兩層找 `.venv/` 與 `Models/`（[LiveTranslator.swift:756](../App/LiveTranslator.swift#L756)）。

### 選用：固定簽章身分

macOS 的錄製授權綁在 App 簽章上。臨時簽章每次編譯都不同，重編後 macOS 會再問一次權限。建一張本機自簽憑證後簽章固定，只需授權一次。

```bash
cd "${REPO_ROOT}"
zsh Scripts/make_signing_identity.sh
```

過程中會出現兩個對話框，各只出現一次：

1. 腳本執行時，macOS 要求輸入 Mac 登入密碼，把憑證設為程式碼簽章可信任
2. 下一次 `build.sh` 簽章時，「codesign 想要使用鑰匙圈中的金鑰」→ 按「永遠允許」

之後 `build.sh` 的輸出會變成「以「LiveTranslator Dev」簽章」。移除：

```bash
zsh "${REPO_ROOT}/Scripts/make_signing_identity.sh" remove
```

## 7. 啟動 App 並授權

```bash
open "${REPO_ROOT}/Build/LiveTranslator.app"
```

第一次啟動 macOS 會詢問螢幕與系統音訊錄製權限：

1. 對話框「LiveTranslator 想要錄製這部電腦的螢幕與音訊」→ 按「允許」
2. 沒有出現對話框或按到拒絕時：系統設定 → 隱私權與安全性 → 螢幕與系統音訊錄製 → 把「LiveTranslator」開關打開
3. 完全結束 App（Cmd+Q）再重新開啟，權限才生效

macOS 26 上第一次啟動時，SpeechAnalyzer 會向 Apple 下載日文與英文的語言資產，需要網路，下載完成前狀態停在「準備中…」。之後辨識完全離線。

底部狀態列會從「準備中…」變成「聆聽中」，模型載入約 10～20 秒。之後播放任何含日文或英文的影片或會議，字幕就出現在視窗裡。

## 8. 操作與逐字稿

底部列由左到右：狀態燈、狀態文字、檔名欄、儲存、終止、暫停／繼續。

| 操作 | 效果 |
|---|---|
| 暫停／繼續 | 暫停只停止聆聽，繼續接著同一段記 |
| 終止 | 結束這一段，檔名欄出現預設檔名（當天日期），點檔名可原地修改 |
| 儲存 | 寫入 `Transcripts/<檔名>.txt`；同名已存在時自動加流水號 |
| 終止後再按播放 | 清空畫面開始新的一段；尚未儲存時先確認 |
| 關閉視窗或 Cmd+Q | 有未儲存內容時用預設檔名自動存一份 |

逐字稿位置：`${REPO_ROOT}/Transcripts/`（不進 Git）。每句兩行，先日文再中文；App 關閉時仍在辨識中的句子標「（未定稿）」：

```
[15:53:34] 必ず誰かを選んでもらうみたいな形にした方がいいですかね。
           是不是应该做成必须选择某人的那种形式呢？
```

## 9. 切換模型

### 辨識引擎

macOS 26 以上預設用 SpeechAnalyzer。要改回 parakeet：先依第 4 節設定 `LIVE_TRANSLATOR_PARAKEET=1` 下載 parakeet 模型，再寫入設定檔並重開 App。

```bash
echo "parakeet" > "${REPO_ROOT}/Models/speech-engine.txt"
```

改回 SpeechAnalyzer：刪掉 `Models/speech-engine.txt`。`Logs/app.log` 每次啟動會記一行 `speech engine: SpeechAnalyzer` 或 `speech engine: parakeet`。

### 翻譯模型

在 `Models/translation-model.txt` 寫入 `Models/` 下的目錄名，重開 App 即生效。沒有這個檔案時依 `Hy-MT2-1.8B-8bit`、`Hunyuan-MT-7B-4bit`、`Hy-MT2-1.8B-4bit`、`Qwen3-4B-4bit` 的順序用第一個已下載的（[LiveTranslator.swift:867](../App/LiveTranslator.swift#L867)）。

```bash
echo "Hy-MT2-7B-4bit" > "${REPO_ROOT}/Models/translation-model.txt"
```

Hy-MT2-7B 4-bit（mlx-community/Hy-MT2-7B-4bit，4.0 GB）翻得比 1.8B 準，代價是每句多約 2 秒、權重多 2.1 GB。

### 選用：定稿後用 Qwen3-ASR 重新辨識

只在 parakeet 模式有效，預設關閉。開啟後每句定稿時把該句音訊交給 Qwen3-ASR-1.7B 重解一次，能把含糊發音校正成通順的字，代價是每句多 1～2 秒、權重多 2.5 GB，16 GB 機器容易 swap。

```bash
cd "${REPO_ROOT}"
uv pip install --python .venv/bin/python "mlx-audio[stt]"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-ASR-1.7B-8bit', local_dir='Models/Qwen3-ASR-1.7B-8bit')"
echo "Qwen3-ASR-1.7B-8bit" > Models/final-model.txt
```

關閉：刪掉 `Models/final-model.txt`。

## 10. 問題排除

| 症狀 | 原因 | 處理 |
|---|---|---|
| 狀態列「尚未完成安裝」 | 找不到 `.venv/bin/python`、模型或 worker | 回到第 2 節；App 是否還在 `Build/` 裡 |
| 狀態列「無法啟動語音辨識」 | SpeechAnalyzer 語言資產下載失敗，通常是第一次啟動時沒有網路 | 連上網路後重開 App |
| 一直「聆聽中」但沒有字幕 | 沒有錄製權限，或重編後簽章變了、授權失效 | 第 7 節重新授權；建立固定簽章（第 6 節） |
| 字幕延遲明顯變長、整台機器變慢 | 記憶體不足開始 swap | 關掉瀏覽器等大程式；活動監視器看「記憶體壓力」與「已使用的交換空間」 |
| 狀態列「處理時發生錯誤」 | worker 例外 | 看 `Logs/app.log` 的最後幾行 |

`Logs/app.log` 記錄每次啟動、狀態切換、按鈕操作與 worker 的錯誤輸出。要看 worker 的解碼細節，帶環境變數啟動：

```bash
open --env LIVE_TRANSLATOR_DEBUG=1 "${REPO_ROOT}/Build/LiveTranslator.app"
```

細節同樣寫進 `Logs/app.log`。
