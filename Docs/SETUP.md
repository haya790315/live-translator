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

底部列由左到右：狀態燈、狀態文字、辨識引擎（Apple／parakeet）、Qwen 重解（只在 parakeet 模式顯示）、潤稿、翻譯模型、檔名欄（終止後出現）、終止、暫停／繼續。

開始一段時就自動建立一個資料夾，逐字稿與錄音放在裡面，三者同名，邊聽邊寫入，不必另外按儲存或錄音：

```text
Transcripts/2026-10-02/
├── 2026-10-02.txt
└── 2026-10-02.wav
```

| 操作 | 效果 |
|---|---|
| 暫停／繼續 | 暫停只停止聆聽與錄音，繼續接著同一段記 |
| 終止 | 結束這一段，檔名欄出現目前檔名；點檔名修改後按 Enter，資料夾、逐字稿、錄音一起改名 |
| 終止後再按播放 | 清空畫面開始新的一段，另建一個新資料夾 |
| 關閉視窗或 Cmd+Q | 寫入最後一次內容後結束 |

命名規則：同名資料夾已存在時自動加流水號（`2026-10-02-2`）。整段沒有任何字幕時，整個資料夾在下一段開始或 App 結束時刪除。

錄音格式為 16 kHz、單聲道、16-bit WAV，約每小時 115 MB。暫停期間不錄，所以錄音長度會比會議實際時間短。錄下的是經過自動增益的音訊（語音拉到約 −23 dBFS，只放大不衰減，尾端軟限幅），也就是辨識模型實際聽到的。擷取不受系統音量與靜音影響，開會時音量轉小或靜音都沒關係。

覺得某場會議錯得特別多時，用下面這支腳本看音訊本身的品質，判斷該怪模型還是怪音訊；怎麼看寫在腳本開頭：

```bash
cd "${REPO_ROOT}"
.venv/bin/python Scripts/audio_qc.py                              # 所有錄音
.venv/bin/python Scripts/audio_qc.py Transcripts/2026-10-05/2026-10-05.wav
```

逐字稿位置：`${REPO_ROOT}/Transcripts/<名稱>/`（不進 Git）。每句兩行，先日文再中文；App 關閉時仍在辨識中的句子標「（未定稿）」：

```
[15:53:34] 必ず誰かを選んでもらうみたいな形にした方がいいですかね。
           是不是应该做成必须选择某人的那种形式呢？
```

## 9. 切換模型

### 辨識引擎

macOS 26 以上預設用 SpeechAnalyzer。兩種引擎都會自動辨別日文與英文。parakeet 模式用 `parakeet-tdt_ctc-0.6b-ja` 辨識日文、`parakeet-tdt-0.6b-v3` 辨識英文，兩個都解一次後挑分數高的。

切換方式：點字幕視窗底列狀態文字右邊的「Apple」或「parakeet」。正在聽時會立刻重啟辨識，已有的字幕保留；暫停或已終止時只記下選擇，下次開始時套用。選擇寫在 `Models/speech-engine.txt`（內容為 `parakeet`；檔案不存在就是 SpeechAnalyzer），重開 App 後仍有效。

切到 parakeet 前要先有模型，沒有的話 App 會跳提示。下載方式：

```bash
cd "${REPO_ROOT}"
LIVE_TRANSLATOR_PARAKEET=1 .venv/bin/python Scripts/download_models.py
```

正在聽時切換，切換當下還在翻譯的那一句不會有中文。`Logs/app.log` 每次啟動會記一行 `speech engine: SpeechAnalyzer` 或 `speech engine: parakeet`。

### 翻譯模型

點字幕視窗底列辨識引擎右邊的模型名稱（例如「Hy-MT2 1.8B」），從選單選擇。選單只列出 `Models/` 下已下載的模型。正在聽時會立刻重啟，已有的字幕保留，但切換當下還在翻譯的那一句不會有中文；暫停或已終止時下次開始才套用。

| 模型 | 大小 | 特性 |
|---|---|---|
| `Hy-MT2-1.8B-8bit` | 1.8 GB | 預設。譯文忠實簡潔，每句 0.5～1.9 秒 |
| `Hy-MT2-7B-4bit` | 4.0 GB | 比 1.8B 準，每句多約 2 秒、記憶體多 2.1 GB |
| `Qwen3-4B-4bit` | 2.1 GB | 通用聊天模型，用系統提示詞要求日譯中 |

選擇寫在 `Models/translation-model.txt`，重開 App 後仍有效。沒有這個檔案，或指定的模型不在時，依上表順序用第一個已下載的（[LiveTranslator.swift](../App/LiveTranslator.swift) 的 `translationModels`）。

`Scripts/download_models.py` 只下載 `Hy-MT2-1.8B-8bit`，另外兩個要自己下載：

```bash
cd "${REPO_ROOT}"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Hy-MT2-7B-4bit', local_dir='Models/Hy-MT2-7B-4bit')"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-4B-4bit', local_dir='Models/Qwen3-4B-4bit')"
```

### 選用：定稿後用 Qwen3-ASR 重新辨識

只在 parakeet 模式有效，預設關閉。開啟後每句定稿時把該句音訊交給 Qwen3-ASR-1.7B 重解一次，能把含糊發音校正成通順的字，代價是每句多 1～2 秒、權重多 2.5 GB，16 GB 機器容易 swap。

```bash
cd "${REPO_ROOT}"
uv pip install --python .venv/bin/python "mlx-audio[stt]"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-ASR-1.7B-8bit', local_dir='Models/Qwen3-ASR-1.7B-8bit')"
```

下載後，在 parakeet 模式下點底列的「Qwen 重解：關／開」切換（Apple 模式下不顯示）。正在聽時會立刻重啟套用，暫停時下次繼續才套用。設定寫在 `Models/final-model.txt`（內容為模型目錄名；檔案不存在就是關閉）。

### 選用：定稿後先潤稿再翻譯

兩種辨識引擎都適用，預設關閉。開啟後每句定稿時：

1. 翻譯前套用術語表：把這場會議中學到的聽錯寫法換成正確寫法（`ショピファイト`→`Shopify`、`sopifi`→`Shopify`），純字串替換，不經過模型
2. 翻譯時 Hy-MT2 的提示詞多帶會議主題與術語表
3. 翻譯完、佇列空著時，Qwen3-4B 潤稿原文：刪語氣詞（えー、あの、um）、刪緊鄰的重講（we can, we can）、刪與上一句結尾重複的開頭、補標點，完成後替換字幕與逐字稿上的原文。只准刪和加標點；模型改了字、加了字、換了語尾，整句不採用。新句子一到就中止潤稿，所以中文不會因為它變慢；代價是講話不停時潤稿常常來不及完成

術語表和會議主題由 Qwen3-4B 在翻譯佇列空著時從最近 8 句原始文字抽出來，每 6 句或閒置 3 秒更新一次。聽錯寫法必須真的出現在逐字稿裡、正確寫法至少兩個字、兩者開頭子音同類（`ショピファイト`→`Shopify` 可以，`ショピファイト`→`LINE` 不行）才收錄。

代價：權重多 2.1 GB。Apple 模式下 worker 共約 4 GB，parakeet 模式下共約 6.4 GB，16 GB 機器開著瀏覽器就會 swap、整體變慢，潤稿會大量超時。和「Qwen 重解」互斥，開一個另一個自動關。

模型與第 9 節翻譯模型清單裡的 `Qwen3-4B-4bit` 是同一份，翻譯模型也選它時共用權重：

```bash
cd "${REPO_ROOT}"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-4B-4bit', local_dir='Models/Qwen3-4B-4bit')"
```

下載後點底列的「潤稿：關／開」切換。正在聽時會立刻重啟套用，暫停時下次繼續才套用。設定寫在 `Models/polish-model.txt`（內容為模型目錄名；檔案不存在就是關閉）。

每句潤稿前後的對照寫在 `Logs/app.log`：

```text
[worker] translate utterance=17 1.12s
[worker] polish utterance=17 1.96s punct '僕の方があまり詳しくない気もしますけど一旦話お伺いできるかなと思います。' -> '僕の方があまり詳しくない気もしますけど、一旦話お伺いできるかなと思います。'
[worker] polish utterance=15 1.07s replace ですか→ません 'いらっしゃらないですかね。' -> None
[worker] polish utterance=18 0.84s preempted '分かりました、じゃあ、ちょっと一回閉めて。' -> None
[worker] memory 2.22s ok topic='讨论Shopify与LINE的整合方案' added=['sopifi=Shopify', 'ショピファイト=Shopify']
```

`ok`／`punct` 是採用；`same` 是模型沒改；`replace`、`insert`、`delete …`、`particle`、`length` 是模型的改動被擋下、沿用原文；`preempted` 是新句子到了、潤稿中止；`timeout` 是超過 6 秒放棄；`skip short` 是短句不送模型。

## 10. 問題排除

| 症狀 | 原因 | 處理 |
|---|---|---|
| 狀態列「尚未完成安裝」 | 找不到 `.venv/bin/python`、模型或 worker | 回到第 2 節；App 是否還在 `Build/` 裡 |
| 狀態列「無法啟動語音辨識」 | SpeechAnalyzer 語言資產下載失敗，通常是第一次啟動時沒有網路 | 連上網路後重開 App |
| 一直「聆聽中」但沒有字幕 | 沒有錄製權限，或重編後簽章變了、授權失效 | 第 7 節重新授權；建立固定簽章（第 6 節） |
| 字幕延遲明顯變長、整台機器變慢 | 記憶體不足開始 swap | 關掉瀏覽器等大程式；活動監視器看「記憶體壓力」與「已使用的交換空間」 |
| 狀態列「處理時發生錯誤」 | worker 例外 | 看 `Logs/app.log` 的最後幾行 |
| 某場會議錯特別多、專有名詞都聽不出來 | 會議軟體送來的音訊本身窄頻（對方麥克風、網路差時 codec 降頻），或系統負載造成掉幀 | `Scripts/audio_qc.py` 看頻寬與斷音（第 8 節）；窄頻要從對方的麥克風與會議設定下手，掉幀則關掉瀏覽器、不要在會議中切模型 |

`Logs/app.log` 記錄每次啟動、狀態切換、按鈕操作與 worker 的錯誤輸出。要看 worker 的解碼細節，帶環境變數啟動：

```bash
open --env LIVE_TRANSLATOR_DEBUG=1 "${REPO_ROOT}/Build/LiveTranslator.app"
```

細節同樣寫進 `Logs/app.log`。
