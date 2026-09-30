# Live Translator

在 Apple Silicon Mac 上完全本地執行的即時「日文／英文 → 中文」字幕工具。日英自動辨別，不需切換；目前輸出簡體中文。

目前已有可編譯的 V1 原型：ScreenCaptureKit 擷取 Mac 系統音訊、MLX 上的 NVIDIA Parakeet 日文與多語（英文）辨識同時運行並按信心值擇一、騰訊 Hy-MT2-1.8B 中文翻譯（帶前兩句上下文）、可拖移的原生浮動字幕。推論流程已用 macOS 合成日文語音跑通並量測，但尚未在真實日文會議音訊與實際的系統音訊擷取上驗證。

已確認的 V1 範圍與驗收方式見 [Docs/V1_PLAN.md](Docs/V1_PLAN.md)。

## 安裝與啟動

在 Apple Silicon Mac（macOS 14+）上，先安裝 `mise`、`uv` 與 Apple Command Line Tools，然後在 repo 根目錄執行：

```bash
zsh Scripts/setup.sh
```

此命令由 `mise` 安裝 Python 3.12，再建立虛擬環境、安裝 MLX 與 onnxruntime 套件、下載約 7 GB 的模型（parakeet-tdt_ctc-0.6b-ja 與 parakeet-tdt-0.6b-v3 各約 2.5 GB、Hy-MT2-1.8B 8-bit 約 1.9 GB、Silero VAD 約 2 MB），從 BSD 商務對話語料合成「句子講完／沒講完」資料並訓練一個字尾 n-gram 斷句分類器（約 1 分鐘），最後用 Apple Swift 工具鏈編譯 `Build/LiveTranslator.app`。完成後打開該 App，便會自動載入模型並開始擷取系統音訊；首次使用需在 macOS 設定允許「螢幕與系統音訊錄製」，若系統要求，授權後重新啟動 App。浮動窗可拖移、可調整大小，內容是一份可捲動的逐句紀錄：正在辨識的日文直接寫在紀錄最後一段（已穩定的部分正常色、可能還會修正的尾巴淡色），句子定稿時文字留在原地，只在下方補上「翻譯中…」再換成中文。底部列顯示狀態燈與暫停／開始按鈕，左上角可關閉或縮小到 Dock，視窗可自由調整大小。

推論期間強制使用本機模型及 Hugging Face 離線模式。底部列有暫停／繼續、終止、儲存三個操作：暫停只停止聆聽，繼續會接著同一段記；終止結束這一段並顯示檔名（預設當天日期，點檔名可原地修改），按儲存寫入 repo 的 `Transcripts/<檔名>.txt`（同名自動加流水號），內容附時間與中文譯文，尚未定稿的日文標「未定稿」；終止後再按播放會清空畫面開始新的一段，尚未儲存時會先確認。關閉 App 時若還有未儲存的內容，會用預設檔名自動存一份。該目錄已被 gitignore。V1 不保存音訊，亦不提供詞彙表。App 目前需保留在此 repo 的 `Build/` 目錄，因為會從同一 repo 讀取 `.venv/` 和 `Models/`。

只改了 Swift 或 Python 程式時，可重新執行 `zsh Scripts/build.sh`，毋須重新下載模型。

每次重新編譯後，macOS 可能把 App 當成新的程式，再問一次「螢幕與系統音訊錄製」權限，因為預設用的是每次都不同的臨時簽章。執行一次 `zsh Scripts/make_signing_identity.sh` 會在登入鑰匙圈建立一張本機自簽憑證「LiveTranslator Dev」（會要求輸入一次 Mac 密碼以信任它），之後 `build.sh` 自動改用這張憑證簽章，重編後權限就會保留。`zsh Scripts/make_signing_identity.sh remove` 可移除。

要切換翻譯模型，在 `Models/translation-model.txt` 寫入模型目錄名（例如 `Hunyuan-MT-7B-4bit`）再重開 App；沒有這個檔案時依序找 `Hy-MT2-1.8B-8bit`、`Hunyuan-MT-7B-4bit`、`Hy-MT2-1.8B-4bit`、`Qwen3-4B-4bit`，用第一個已下載的。狀態列會顯示實際載入的模型名稱。Hy-MT2 與 Qwen 會帶前兩句日文作為上下文整句翻譯；第一代 Hunyuan-MT-7B 的提示格式固定，只能逐句獨立翻。

## 目標

```text
macOS system audio
        ↓
ScreenCaptureKit
        ↓
streaming Japanese STT
        ↓
Traditional Chinese translation
        ↓
floating live subtitles
```

- 目標平台：Apple Silicon（目前開發／測試機為 M1、16 GB unified memory）
- 音訊來源：macOS system audio，不以麥克風為主要輸入
- 音訊擷取：ScreenCaptureKit
- STT 採用 NVIDIA parakeet-tdt_ctc-0.6b-ja（MLX 移植，parakeet-mlx）
  - 先前為 Whisper large-v3-turbo；日文評測中 Parakeet 在 JSUT／Common Voice 更準且快數十倍
  - 選用、預設關閉：句子定稿後交給 Qwen3-ASR-1.7B 8-bit（mlx-audio）重新辨識該句音訊。它帶語言模型解碼器，能把含糊發音校正成通順的字並自動判定日／英，但每句多 1～2 秒、多佔 2.5 GB 記憶體，16 GB 機器上容易 swap。要開啟：`uv pip install --python .venv/bin/python "mlx-audio[stt]"`，把 `mlx-community/Qwen3-ASR-1.7B-8bit` 下載到 `Models/Qwen3-ASR-1.7B-8bit`，再在 `Models/final-model.txt` 寫入 `Qwen3-ASR-1.7B-8bit` 後重開 App
- 翻譯目前採用：Hy-MT2-1.8B 8-bit（騰訊第二代翻譯模型，Apache-2.0，支援指令式上下文）
  - 先前為 Qwen3-4B 4-bit 與 Hunyuan-MT-7B 4-bit；備選 Hy-MT2-7B、Qwen3.5-9B、TranslateGemma
- 低延遲 baseline：NLLB-200 distilled 600M
- 輸出：日文即時逐字稿加定稿後的中文字幕（目前為簡體中文）

## 設計原則

- 全程本地處理，會議音訊與文字預設不離開裝置。
- 先量測、再選 backend；不先假設 MLX 或 whisper.cpp 一定較快。
- 以「使用者多久看到正確字幕」為主要 KPI，而非單一模型 benchmark 分數。
- 專有名詞、產品名與中英日混用需納入測試。
- M1 16 GB 同一時間只載入一個翻譯候選模型，避免不必要的 memory pressure。

## 預計元件

| 元件 | 初始方案 | 備註 |
|---|---|---|
| System audio capture | ScreenCaptureKit | 原生 Swift/macOS app |
| Audio preprocessing | 16 kHz mono PCM + Silero VAD | ScreenCaptureKit 直接輸出 16 kHz 單聲道；VAD 以 onnxruntime 在 CPU 跑，每 32 ms 一幀 |
| Streaming STT | parakeet-tdt_ctc-0.6b-ja 加 parakeet-tdt-0.6b-v3 | 兩個模型每次都解碼，依信心值、輸出長度與片假名比例擇一；v3 缺席時只跑日文 |
| Final STT（選用） | Qwen3-ASR-1.7B 8-bit | 預設關閉。開啟後句子定稿時重解該句音訊；只採用與 parakeet 結果有重疊的句子，避免混入前後句；不足 2 秒的短句沿用 parakeet 判定的語言 |
| Translation | Hy-MT2-1.8B 8-bit | 整句翻譯、帶前兩句上下文、提示前綴 KV cache 重用、greedy 解碼 |
| Translation baseline | NLLB-200 distilled 600M | 延遲基準；授權限制見下方 |
| Subtitle UI | AppKit floating panel | 已有可拖移原型，待真實會議驗證 |

## 目錄與檔案

```text
App/LiveTranslator.swift          macOS 原生 App：ScreenCaptureKit 擷取系統音訊、浮動字幕視窗、啟動 Python worker、儲存逐字稿
App/Info.plist                    App 設定與螢幕錄製權限說明
App/AppIcon.icns                  App 圖示
Inference/worker.py               推論 worker：VAD、parakeet 辨識、Hy-MT2 翻譯（選用的 Qwen3-ASR 定稿重解）；以 JSON 行與 App 溝通
Scripts/setup.sh                  一鍵安裝：建 venv、裝套件、下載模型、訓練斷句分類器、編譯 App
Scripts/download_models.py        從 Hugging Face 下載模型到 Models/
Scripts/make_segment_data.py      下載 BSD 語料並合成「句子講完／沒講完」訓練資料
Scripts/train_segmenter.py        訓練字尾 n-gram 斷句分類器，輸出 Models/segmenter/weights.json
Scripts/build.sh                  編譯並簽章 Build/LiveTranslator.app（改了程式後重跑）
Scripts/make_signing_identity.sh  選用：建立本機自簽憑證，重編後不必重新授權
Docs/V1_PLAN.md                   設計紀錄、量測結果與已知風險
PREREQUISITES.md                  環境、權限與模型授權的前置說明
mise.toml                         指定 Python 3.12
Models/                           本地模型與訓練產物（內容不進 Git）
Build/、Transcripts/、Logs/        編譯產物、逐字稿、執行紀錄（皆不進 Git）
```

所有推論都在本機完成。網路只用於安裝時下載套件與模型（Hugging Face 與 GitHub 上的 BSD 語料）；執行時 worker 強制離線模式。

## 開始之前

請先閱讀 [PREREQUISITES.md](PREREQUISITES.md)。準備一段已獲授權的 30～60 秒日文會議音訊，配合人工校對原文與譯文驗收。`Scripts/setup.sh` 只下載目前使用的模型，不下載整個候選矩陣。

## 成功指標（初始目標）

- 日文 partial transcript：500～800 ms 內可見
- 繁中 partial subtitle：約 1 秒內可見
- 穩定句子：說完後約 1～2 秒完成
- 30 分鐘測試期間不發生 swap storm、持續掉速或 app 無回應
- 專有名詞可由 glossary 穩定保留或正規化

以上是產品目標，不是尚未量測的保證值。

## 授權提醒

- parakeet-tdt_ctc-0.6b-ja 與 parakeet-tdt-0.6b-v3 權重標示為 CC-BY-4.0；Silero VAD 為 MIT。
- Qwen3-ASR-1.7B（選用）權重標示為 Apache-2.0，MLX 8-bit 版由 mlx-community 轉換。
- 斷句分類器的訓練資料合成自 BSD（Business Scene Dialogue）語料，該語料為 CC BY-NC-SA 4.0，只適合私人、非商用；`Models/segment-data/` 內同時有 Laya 微調格式的 JSONL。
- Hy-MT2-1.8B 權重標示為 Apache-2.0，MLX 8-bit 版由 mlx-community 轉換。
- Hunyuan-MT-7B（備選）採騰訊 Hunyuan 社群授權，商用有條件限制；MLX 4-bit 版由第三方（shawizir）轉換。
- `facebook/nllb-200-distilled-600M` 權重標示為 CC-BY-NC-4.0，僅適合作為研究／非商用 baseline；不可直接假設可用於商業產品。
- 正式發佈前必須再次審查所有模型、程式庫及測試音訊的授權。

## 參考資料

- [Apple ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit)
- [Apple：Capturing screen content in macOS](https://developer.apple.com/documentation/screencapturekit/capturing-screen-content-in-macos)
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [NLLB-200 distilled 600M model card](https://huggingface.co/facebook/nllb-200-distilled-600M)
