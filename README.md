# Live Translator

在 Apple Silicon Mac 上完全本地執行的即時「日文 → 繁體中文」字幕工具。

目前專案只包含骨架與前置要件；尚未實作音訊擷取、語音辨識、翻譯或字幕 UI。

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
- STT 首選：Whisper large-v3-turbo
  - 待比較 MLX 與 whisper.cpp backend
- 翻譯首選：Qwen 3B／4B、4-bit
  - 第一候選為 Qwen3-4B 4-bit
  - 若 M1 16 GB 上延遲或記憶體不理想，再測 3B 級模型
- 低延遲 baseline：NLLB-200 distilled 600M
- 輸出：可更新 partial result 的繁體中文字幕

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
| Audio preprocessing | 16 kHz mono PCM + VAD | 實際 resampling/chunk 策略待測 |
| Streaming STT | Whisper large-v3-turbo | MLX vs whisper.cpp |
| Translation | Qwen 3B/4B 4-bit | 短 context、non-thinking、限制只輸出譯文 |
| Translation baseline | NLLB-200 distilled 600M | 延遲基準；授權限制見下方 |
| Subtitle UI | SwiftUI/AppKit floating panel | 尚未實作 |

## 目錄

```text
App/          macOS 原生 app、ScreenCaptureKit 與字幕 UI
Inference/    STT／翻譯 backend 與模型介面
Benchmarks/   音訊樣本規格、量測工具與結果
Docs/         架構與設計紀錄
Scripts/      開發、下載與 benchmark 輔助腳本
Tests/        自動化測試
Models/       本地模型目錄（內容不進 Git）
```

## 開始之前

請先閱讀 [PREREQUISITES.md](PREREQUISITES.md)。目前不要一次下載所有模型；先準備一段已獲授權的 30～60 秒日文會議音訊，再逐一跑候選方案。

## 成功指標（初始目標）

- 日文 partial transcript：500～800 ms 內可見
- 繁中 partial subtitle：約 1 秒內可見
- 穩定句子：說完後約 1～2 秒完成
- 30 分鐘測試期間不發生 swap storm、持續掉速或 app 無回應
- 專有名詞可由 glossary 穩定保留或正規化

以上是產品目標，不是尚未量測的保證值。

## 授權提醒

- Qwen3-4B 權重標示為 Apache-2.0。
- `facebook/nllb-200-distilled-600M` 權重標示為 CC-BY-NC-4.0，僅適合作為研究／非商用 baseline；不可直接假設可用於商業產品。
- 正式發佈前必須再次審查所有模型、程式庫及測試音訊的授權。

## 參考資料

- [Apple ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit)
- [Apple：Capturing screen content in macOS](https://developer.apple.com/documentation/screencapturekit/capturing-screen-content-in-macos)
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [NLLB-200 distilled 600M model card](https://huggingface.co/facebook/nllb-200-distilled-600M)
