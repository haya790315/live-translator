# 前置要件

這份文件記錄 prototype 階段的保守基線。版本會在第一次 benchmark 後鎖定；目前不提供一鍵安裝，也不下載任何模型。

## 1. 硬體與作業系統

### 必要

- Apple Silicon Mac（arm64）
- 目前目標機：M1、16 GB unified memory
- macOS 14 Sonoma 或更新版本
- 至少 20 GB 可用磁碟空間，建議 30 GB，方便同時保留模型、套件 cache、測試音訊與 benchmark 結果

### 建議

- macOS 15 或更新版本：較新的 MLX 記憶體處理及 Apple 範例環境較一致
- 測試時關閉高記憶體占用程式，並觀察 Activity Monitor 的 Memory Pressure 與 Swap Used
- 電源接上，避免長時間測試受低耗電模式影響

> 最新 MLX 原始碼目前要求 Python 3.10+，且其 Metal build baseline 為 macOS 14 SDK／macOS 14。專案先採 macOS 14+，實際 deployment target 仍需以安裝測試確認。

## 2. Apple 開發工具

- Xcode 16 或更新版本
- Xcode Command Line Tools
- Swift（隨 Xcode 提供）
- macOS SDK（隨 Xcode 提供）

確認方式：

```bash
xcodebuild -version
xcrun swift --version
xcrun --sdk macosx --show-sdk-version
```

ScreenCaptureKit 會提供 system audio 的 `CMSampleBuffer`。初版預計由 Swift app 收流，再轉成 STT 所需的 16 kHz mono PCM。是否採跨程序 IPC、嵌入式 runtime 或純 Swift/C++ 整合，留待 benchmark 後決定。

## 3. Python 與原生 build 工具

### Python

- 原生 arm64 Python 3.11 或 3.12（prototype 建議）
- 隔離環境：`venv` 或 `uv`
- 不要用經 Rosetta 執行的 x86_64 Python

確認架構：

```bash
python3 -c "import platform; print(platform.machine())"
```

預期輸出為 `arm64`。

### 預計需要

- Git
- Homebrew（Apple Silicon 預設安裝路徑）
- CMake（whisper.cpp build）
- FFmpeg（測試音訊轉成 16 kHz mono PCM，也為 MLX Whisper 的常用前置工具）
- Ninja（可選，縮短 C/C++ build 時間）
- `pkg-config`（可選，原生相依套件偵測）

預計安裝命令（尚未執行）：

```bash
brew install cmake ffmpeg ninja pkg-config
```

MLX 路線預計會用到 `mlx-whisper`；Qwen 路線優先評估 `mlx-lm`。NLLB baseline 可能使用 Transformers/PyTorch、ONNX Runtime 或轉換後格式，必須先以 M1 latency 與 peak memory 決定，暫不鎖套件版本。

## 4. macOS 權限

ScreenCaptureKit 不會在未授權時靜默擷取內容。第一次啟動時需允許：

- 系統設定 → 隱私權與安全性 → 螢幕與系統音訊錄製
  - macOS 版本不同時，名稱可能顯示為「螢幕錄製」或相近文字
- app target 的 `Info.plist` 需提供 `NSScreenCaptureUsageDescription`

可能需要完全結束並重新開啟 app 才會套用權限。若日後加入麥克風模式，還需 `NSMicrophoneUsageDescription` 與麥克風權限；純 system audio prototype 不應要求不必要的麥克風權限。

實作時：

- `SCStreamConfiguration.capturesAudio = true`
- 視需要設定 `excludesCurrentProcessAudio = true`，防止字幕 app 自己的提示音回灌
- 優先使用 Apple 建議的 system content picker，讓使用者選擇允許擷取的顯示器／app／視窗
- 明確顯示正在擷取的狀態，且提供立即停止按鈕

## 5. 模型下載、磁碟與 RAM 預估

數字是規劃值，不等於實測 peak resident memory。模型載入後還有 KV cache、運算 buffer、音訊 ring buffer、runtime 與 UI 開銷。

| 候選 | 下載／權重約略大小 | Prototype 工作記憶體規劃 | 備註 |
|---|---:|---:|---|
| MLX Whisper large-v3-turbo | 約 1.6 GB | 約 2～4 GB | MLX Community 目前檔案約 1.61 GB |
| whisper.cpp large-v3-turbo F16 | 約 1.5～1.6 GB | 約 2～4 GB | 可另測 q5_0 約 0.55～0.6 GB |
| Qwen3-4B 4-bit | 約 2.3～2.6 GB | 約 3～5 GB | 短 context、non-thinking，控制 KV cache |
| Qwen 3B 級 4-bit | 約 1.8～2.2 GB（待選 checkpoint） | 約 2.5～4 GB | 作為 M1 延遲／記憶體 fallback |
| NLLB-200 distilled 600M | 約 2.5 GB（原始 checkpoint） | 約 3～5 GB | 量化／ONNX 大小依格式而異；非商用授權 |

### M1 16 GB 容量策略

- Qwen 路線預估整體常駐約 8～12 GB，需以實測 peak memory 為準。
- benchmark 時不要同時載入 Qwen 與 NLLB。
- 避免過長 context；翻譯只帶最近 2～3 句與精簡 glossary。
- 若發生高 memory pressure 或 swap，依序測：縮短 context、較小／量化 STT、3B 翻譯模型、NLLB baseline。
- 模型放在 `Models/` 或外部 cache；都不提交 Git。

## 6. 測試資料

準備至少一段 30～60 秒、可合法使用的日文會議音訊，內容最好包含：

- 工程會議口語與不完整句子
- 中英日混合詞彙
- 人名、公司名、產品名與 glossary 詞彙
- 靜音、搶話、背景音與不同語速

另準備人工校對過的日文逐字稿及理想繁中翻譯，才能同時比較速度與品質。真實會議音訊不得在未獲參與者同意時加入 repo 或外傳。

## 7. 後續 benchmark 矩陣

### 音訊擷取與切段

- ScreenCaptureKit callback 穩定性、sample rate／channel format
- 48 kHz stereo → 16 kHz mono 的轉換成本
- chunk 大小與 overlap：例如 0.5、1.0、2.0 秒
- VAD 開／關及靜音時資源使用量
- 長時間擷取、裝置睡眠／喚醒、權限撤銷與來源結束

### STT

- MLX Whisper large-v3-turbo vs whisper.cpp large-v3-turbo
- whisper.cpp F16 vs q5_0（必要時再加 q8_0）
- partial 更新頻率、穩定度、重複字與回改幅度
- 日文 WER/CER、專有名詞正確率
- cold start、warm start、real-time factor、p50/p95 latency、peak RAM、平均 CPU/GPU
- 若 large-v3-turbo 不穩，再測 medium／distilled 類候選

### 翻譯

- Qwen3-4B 4-bit vs 選定的 3B 4-bit 候選 vs NLLB-200 distilled 600M
- 完整句與 partial sentence 的品質差異
- 有／無最近 2～3 句 context
- 有／無 glossary 與 protected-token 流程
- Qwen non-thinking 模式、最大輸出長度與 prompt 長度
- 首 token latency、完成 latency、tokens/s、peak RAM
- 幻覺、漏譯、重複、繁簡轉換錯誤與專有名詞保留率

### End-to-end

- 音訊抵達 → 日文 partial
- 音訊抵達 → 繁中 partial
- 句尾 → 穩定繁中字幕
- 30 分鐘連續會議的 p50/p95/p99 latency、memory growth、thermal throttling 與 swap
- 字幕更新是否閃爍、回改是否容易閱讀
- app 本身音訊是否被排除、耳機／喇叭切換是否正常

每次結果至少記錄：macOS、Xcode、Python、backend commit／套件版本、模型精確 revision、量化格式、prompt、context 長度、音訊樣本與電源狀態。

## 8. Go / No-Go 判準

在 M1 16 GB 上，初步 Go 條件：

- 繁中 partial 大致維持在約 1 秒，穩定句約 1～2 秒
- 連續 30 分鐘無明顯 memory leak 或持續增長的 swap
- 專有名詞與工程語境品質足以跟上會議
- 不需雲端服務即可完成主要流程

若未達標，優先降低模型／context 成本，再考慮改變整體架構。

## 9. 重要授權限制

`facebook/nllb-200-distilled-600M` model card 標示 CC-BY-NC-4.0，且將此模型描述為研究用途、非 production deployment。它在本專案中僅是非商用低延遲 baseline。若產品有商業用途，需改用授權相容的翻譯模型或另行取得權利。
