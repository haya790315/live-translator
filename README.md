# Live Translator

在 Apple Silicon Mac 上完全本地執行的即時字幕工具：擷取 Mac 正在播放的系統音訊，辨識日文與英文（自動辨別，不需切換），翻成簡體中文，顯示在可拖移的浮動視窗，並可把逐字稿存成文字檔。音訊與文字不離開這台機器。

```text
macOS 系統音訊（ScreenCaptureKit）
        ↓
Apple SpeechAnalyzer 日文與英文轉錄器同時辨識，依結果挑語言（macOS 內建，本機執行；術語表作為 contextual strings）
        ↓
規則刪贅詞（えっと、えー、um…）＋ 術語表替換（Models/glossary.txt）
        ↓
Hy-MT2-1.8B 中文翻譯（MLX，帶前兩句上下文與術語表）
        ↓
（選用）Qwen3-4B 在空檔補標點、學會議中的術語與主題；16 GB 機器開會時不建議開
        ↓
浮動字幕視窗、Transcripts/*.txt
```

已在真實線上會議中使用。設計取捨、量測結果與已知風險見 [Docs/V1_PLAN.md](Docs/V1_PLAN.md)。

## 安裝與啟動

需要 macOS 26 以上的 Apple Silicon Mac、16 GB 記憶體、5 GB 磁碟，以及 Xcode Command Line Tools、Homebrew、`mise`、`uv`。macOS 14 到 25 改用 parakeet 辨識，安裝時會多下載 4.6 GB 模型。

1. 取得原始碼

   ```bash
   git clone https://github.com/haya790315/live-translator.git
   cd live-translator
   ```

2. 一鍵安裝：建立 Python 環境、下載翻譯模型 1.8 GB、訓練斷句分類器、編譯 App

   ```bash
   zsh Scripts/setup.sh
   ```

3. 打開 `Build/LiveTranslator.app`，第一次啟動時允許「螢幕與系統音訊錄製」，然後完全結束 App 再重開
4. 播放任何含日文或英文的影片或會議，字幕就會出現

每一步的確認方式、期待輸出、授權畫面的位置、操作說明與問題排除都在 [Docs/SETUP.md](Docs/SETUP.md)。只改了程式要重編：`zsh Scripts/build.sh`。

## 目錄與檔案

```text
App/LiveTranslator.swift          macOS 原生 App：ScreenCaptureKit 擷取系統音訊、自動增益、浮動字幕視窗、啟動 Python worker、儲存逐字稿
App/SpeechEngine.swift            SpeechAnalyzer 辨識：日英兩個轉錄器、切句定稿、挑語言、正式定稿後的修正
App/Info.plist                    App 設定與螢幕錄製權限說明
App/AppIcon.icns                  App 圖示
Inference/worker.py               推論 worker：Hy-MT2 翻譯；選用的 Qwen3-4B 潤稿與會議記憶；parakeet 模式下另負責 VAD 與辨識；以 JSON 行與 App 溝通
Scripts/setup.sh                  一鍵安裝：建 venv、裝套件、下載模型、訓練斷句分類器、編譯 App
Scripts/download_models.py        從 Hugging Face 下載模型到 Models/
Scripts/make_segment_data.py      下載 BSD 語料並合成「句子講完／沒講完」訓練資料
Scripts/train_segmenter.py        訓練字尾 n-gram 斷句分類器，輸出 Models/segmenter/weights.json
Scripts/build.sh                  編譯並簽章 Build/LiveTranslator.app（改了程式後重跑）
Scripts/make_signing_identity.sh  選用：建立本機自簽憑證，重編後不必重新授權
Scripts/audio_qc.py               檢查錄音的音量、頻寬、斷音，判斷辨識錯誤是不是音訊本身的問題
Docs/SETUP.md                     開發環境建置、模型下載、啟動、操作與問題排除
Docs/V1_PLAN.md                   設計紀錄、量測結果與已知風險
mise.toml                         指定 Python 3.12
Models/                           本地模型、訓練產物與 glossary.txt 術語表（內容不進 Git）
Build/、Transcripts/、Logs/        編譯產物、逐字稿、執行紀錄（皆不進 Git）
```

網路只用於安裝時下載套件與模型（Hugging Face，以及 GitHub 上的 BSD 語料）；執行時 worker 強制離線模式。

## 授權

- 語音辨識使用 macOS 內建的 SpeechAnalyzer，語言資產由 Apple 提供。parakeet 模式用的 parakeet-tdt_ctc-0.6b-ja 與 parakeet-tdt-0.6b-v3 權重為 CC-BY-4.0；Silero VAD 為 MIT。
- Hy-MT2-1.8B 權重為 Apache-2.0，MLX 8-bit 版由 mlx-community 轉換。選用的 Qwen3-ASR-1.7B 與 Qwen3-4B 同為 Apache-2.0。
- 斷句分類器的訓練資料合成自 BSD（Business Scene Dialogue）語料，該語料為 CC BY-NC-SA 4.0，因此整體只適合私人、非商用。
- 正式發佈前須再次審查所有模型、程式庫及測試音訊的授權。
