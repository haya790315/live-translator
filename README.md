# Live Translator

在 Apple Silicon Mac 上完全本地執行的即時字幕工具：擷取 Mac 正在播放的系統音訊，辨識日文與英文（自動辨別，不需切換），翻譯成繁體中文（或日英互譯），顯示在可拖移的浮動視窗，並把逐字稿存成文字檔。音訊與文字不離開這台機器。

```text
macOS 系統音訊（ScreenCaptureKit）
        ↓
Apple SpeechAnalyzer 日文與英文轉錄器同時辨識，即時顯示初稿並切句（macOS 內建；術語表作為 contextual strings）
        ↓
每句定稿後，Qwen3-ASR-1.7B 指定日文重解該句音訊（MLX；英文照原文輸出；1.5 秒以下的日文短句沿用初稿）
        ↓
規則刪贅詞（えっと、えー、um…）＋ 術語表替換（Models/glossary.txt）
        ↓
（選用，預設關閉）翻譯前潤稿：Qwen3-4B 刪贅詞、補標點、修正聽錯的術語，輸出仍是原文語言
        ↓
PLaMo 3 Translate 翻譯（前兩句原文當前文、術語表當 glossary）
        ↓
浮動字幕視窗、Transcripts/*.txt
```

譯文選單有兩個選項：「繁體中文」（日文、英文都翻成繁體中文，預設）與「日英互譯」（日→英、英→日）。翻譯模型預設 PLaMo 2B，可換 8B（較準、較慢）。沒有下載翻譯模型時只出逐字稿。

已在真實線上會議中使用（逐字稿部分）。設計取捨、量測結果與已知風險見 [Docs/V1_PLAN.md](Docs/V1_PLAN.md)（為初版規劃：parakeet、Hy-MT2，已不是現況）。

## 安裝與啟動

需要 macOS 26 以上的 Apple Silicon Mac、16 GB 記憶體、7 GB 磁碟，以及 Xcode Command Line Tools、Homebrew、`mise`、`uv`。記憶體用量：Qwen3-ASR 約 2.5 GB、PLaMo 2B 約 2.6 GB；換 PLaMo 8B 約 6 GB；開潤稿再加 Qwen3-4B 約 2.4 GB。

1. 取得原始碼

   ```bash
   git clone https://github.com/haya790315/live-translator.git
   cd live-translator
   ```

2. 一鍵安裝：建立 Python 環境、下載 Qwen3-ASR 2.3 GB 與 PLaMo 2B 2.3 GB、編譯 App

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
Inference/worker.py               推論 worker：Qwen3-ASR 重解每句、規則刪贅詞與術語表、選用的翻譯前潤稿、PLaMo 3 Translate 翻譯；以 JSON 行與 App 溝通
Scripts/setup.sh                  一鍵安裝：建 venv、裝套件、下載模型、編譯 App
Scripts/download_models.py        從 Hugging Face 下載模型到 Models/
Scripts/build.sh                  編譯並簽章 Build/LiveTranslator.app（改了程式後重跑）
Scripts/make_signing_identity.sh  選用：建立本機自簽憑證，重編後不必重新授權
Scripts/audio_qc.py               檢查錄音的音量、頻寬、斷音，判斷辨識錯誤是不是音訊本身的問題
Docs/SETUP.md                     開發環境建置、模型下載、啟動、操作與問題排除
Docs/V1_PLAN.md                   初版（parakeet、Hy-MT2）的設計紀錄與量測，已不是現況
mise.toml                         指定 Python 3.12
Models/                           本地模型與 glossary.txt 術語表（內容不進 Git）
Build/、Transcripts/、Logs/        編譯產物、逐字稿、執行紀錄（皆不進 Git）
```

網路只用於安裝時下載套件與模型（Hugging Face）；執行時 worker 強制離線模式。

## 授權

- 語音辨識使用 macOS 內建的 SpeechAnalyzer，語言資產由 Apple 提供。選用的 Qwen3-ASR-1.7B 為 Apache-2.0。
- 翻譯使用 Preferred Networks 的 PLaMo 3 Translate，適用 [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en)：個人與非商用免費；商用且年營收 10 億日圓以下需先向 Preferred Networks 登記，營收超過 10 億日圓需另外簽約。
- 載入 PLaMo 時會執行模型 repo 附的 `modeling_plamo_mlx.py`（`trust_remote_code`）。`Scripts/download_models.py` 把 revision 釘在檢視過的版本，換版本前請先檢視新版程式碼。
- 正式發佈前須再次審查所有模型、程式庫及測試音訊的授權。
