# 開發環境建置與啟動

在一台 Apple Silicon Mac 上從零裝好、編譯並啟動 App 的步驟。只改了程式要重編，看第 6 節；只想切換譯文或模型，看第 9 節。

## 前提

| 項目 | 要求 |
|---|---|
| 硬體 | Apple Silicon Mac；記憶體 16 GB 以上（Qwen3-ASR 約 2.5 GB、PLaMo 2B 約 2.6 GB；換 PLaMo 8B 約 6 GB） |
| macOS | 26 以上（辨識用系統內建的 SpeechAnalyzer，`Info.plist` 的 `LSMinimumSystemVersion` 是 26.0） |
| 磁碟 | 至少 7 GB（PLaMo 2B 2.3 GB、Qwen3-ASR 2.3 GB、Python 環境約 2 GB）；PLaMo 8B 再加約 6 GB |
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

第 3 到 4 節與第 6 節全部由一個腳本執行；想分步驟做或只重做其中一步，再看各節。

```bash
cd "${REPO_ROOT}"
zsh Scripts/setup.sh
```

**期待輸出**（最後一行）

```
安裝完成。打開 Build/LiveTranslator.app 即可開始。
```

所需時間主要在下載模型：約 4.6 GB，每秒 2.5 MB 的連線約 30 分鐘。

## 3. 建立 Python 環境

```bash
cd "${REPO_ROOT}"
mise install python
uv venv --python "$(mise which python)" .venv
uv pip install --python .venv/bin/python numpy numba mlx-lm "mlx-audio[stt]" huggingface_hub hf_xet
```

確認：

```bash
"${REPO_ROOT}/.venv/bin/python" -c "import platform, mlx.core, mlx_lm, numba; print(platform.machine(), mlx.core.__version__)"
```

**期待輸出**

```
arm64 0.32.3
```

第一個值必須是 `arm64`。經 Rosetta 執行的 x86_64 Python 裝不了 MLX。`numba` 是 PLaMo 模型程式需要的，`transformers` 由 `mlx-lm` 一併裝進來。

## 4. 下載模型

模型放在 `Models/` 下，目錄名固定，App 依目錄名尋找。即時辨識由系統內建的 SpeechAnalyzer 負責，每句定稿再交給 Qwen3-ASR 重解，然後由 PLaMo 3 Translate 翻譯。第一次下載時會順便寫入 `Models/final-model.txt`，讓「Qwen 重解」預設開啟。

| 目錄（`Models/` 下） | 用途 | 大小 | 何時下載 | 來源 | 授權 |
|---|---|---:|---|---|---|
| `plamo-3-translate-2b-MLX-4bit` | 翻譯（預設） | 2.3 GB | 一律 | pfnet/plamo-3-translate-2b-MLX-4bit | PLaMo Community License |
| `Qwen3-ASR-1.7B-8bit` | 每句定稿後重解（預設開啟） | 2.3 GB | 一律 | mlx-community/Qwen3-ASR-1.7B-8bit | Apache-2.0 |
| `Qwen3-4B-4bit` | 翻譯前潤稿（選用，預設關閉） | 2.3 GB | 設定 `LIVE_TRANSLATOR_POLISH=1` | mlx-community/Qwen3-4B-4bit | Apache-2.0 |
| `plamo-3-translate-8b-MLX-4bit` | 翻譯（較準、較慢，選用） | 約 6 GB | 設定 `LIVE_TRANSLATOR_PLAMO_8B=1` | pfnet/plamo-3-translate-8b-MLX-4bit | PLaMo Community License |

```bash
cd "${REPO_ROOT}"
HF_HUB_DISABLE_XET=1 .venv/bin/python Scripts/download_models.py
```

想要 8B 時，加上 `LIVE_TRANSLATOR_PLAMO_8B=1` 再下載（已下載的 2B 與 Qwen3-ASR 會略過）：

```bash
cd "${REPO_ROOT}"
LIVE_TRANSLATOR_PLAMO_8B=1 HF_HUB_DISABLE_XET=1 .venv/bin/python Scripts/download_models.py
```

`HF_HUB_DISABLE_XET=1` 讓 Hugging Face 用一般 HTTP 下載，避免 xet 傳輸卡住。中斷後重跑同一命令會接著下載。

> **安全注意**：PLaMo 的 MLX 版沒有內建在 `mlx-lm` 裡，載入時會執行模型 repo 附的 `modeling_plamo_mlx.py`（`trust_remote_code`）。`download_models.py` 把 revision 釘在檢視過的版本（2B 是 `63777ddc…`，8B 是 `9d876844…`），不會跟著 main 更新；要換版本時先檢視新版的程式碼再改腳本。

確認：

```bash
ls "${REPO_ROOT}"/Models/*/config.json
```

**期待輸出**

```
.../Models/Qwen3-ASR-1.7B-8bit/config.json
.../Models/plamo-3-translate-2b-MLX-4bit/config.json
```

## 5. 授權（PLaMo Community License）

翻譯模型 PLaMo 3 Translate 以 PLaMo Community License 提供，使用前請閱讀條款：https://plamo.preferredai.jp/info/plamo-community-license-en

- 個人與非商用免費。
- 商用且年營收 10 億日圓以下：需要先向 Preferred Networks 登記。
- 營收超過 10 億日圓：需要另外簽約。

其他元件的授權列在 README。

## 6. 編譯 App

```bash
cd "${REPO_ROOT}"
zsh Scripts/build.sh
```

**期待輸出**

```
已建立 Build/LiveTranslator.app（臨時簽章；執行 zsh Scripts/make_signing_identity.sh 可固定簽章，重編後不必再授權）
```

腳本先編到暫存目錄再換名，App 開著的時候重編也不會被殺掉。App 必須留在 `Build/` 裡，因為它從自己的位置往上兩層找 `.venv/` 與 `Models/`（[LiveTranslator.swift:1172](../App/LiveTranslator.swift#L1172)）。

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

第一次啟動時，SpeechAnalyzer 會向 Apple 下載日文與英文的語言資產，需要網路，下載完成前狀態停在「準備中…」。之後辨識完全離線。

底部狀態列會從「準備中…」變成「聆聽中」，模型載入約 10～20 秒。之後播放任何含日文或英文的影片或會議，字幕就出現在視窗裡。

## 8. 操作與逐字稿

底部列由左到右：狀態燈、狀態文字、Qwen 重解、譯文（選單）、檔名欄（終止後出現）、終止、暫停／繼續。

字幕先出現 SpeechAnalyzer 的初稿，每句講完約 1～2.5 秒後換成 Qwen3-ASR 的定稿，再約 1 秒後出現譯文。輸入語言（日文或英文）自動判斷，不需切換。

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

逐字稿位置：`${REPO_ROOT}/Transcripts/<名稱>/`（不進 Git）。每句兩行，先原文再譯文（沒有安裝翻譯模型時只有原文）；App 關閉時仍在辨識中的句子標「（未定稿）」：

```
[15:53:34] 必ず誰かを選んでもらうみたいな形にした方がいいですかね。
           是不是應該做成一定要讓人選出某個人的形式比較好呢？
```

## 9. 切換譯文與模型

### 譯文

點底列的「譯文：…」，選單分兩區。

| 選項 | 效果 |
|---|---|
| 繁體中文（預設） | 日文、英文原文都翻成繁體中文 |
| 日英互譯 | 日文翻成英文，英文翻成日文 |

選擇寫在 `Models/translation-target.txt`（內容 `zh-hant` 或 `ja-en`；檔案不存在或內容不認得就是 `zh-hant`），重開 App 後仍有效。

### 翻譯模型

同一個選單的「翻譯模型」區只列出 `Models/` 下已下載的 PLaMo。

| 模型 | 記憶體 | 特性 |
|---|---|---|
| `plamo-3-translate-2b-MLX-4bit` | 約 2.6 GB | 預設。每句約 0.8～1.3 秒（基本款 M1） |
| `plamo-3-translate-8b-MLX-4bit` | 約 6 GB（多約 3.7 GB） | 較準、較慢 |

選擇寫在 `Models/translation-model.txt`，重開 App 後仍有效。檔案內容不是 `plamo` 開頭（例如舊版留下的 `Hy-MT2-1.8B-8bit`）或指定的模型沒下載時，用第一個已下載的 PLaMo。完全沒有 PLaMo 時，按鈕顯示「譯文：未安裝」，只出逐字稿。

正在聽時切換譯文、翻譯模型或 Qwen 重解，App 會先收尾再切換：停止擷取、讓還沒定稿的句子定稿，等舊 worker 把已送出的句子重解、翻譯完並結束，才用新設定啟動新 worker（狀態列顯示「切換中…」）。已有的字幕與切換前講的句子都保留，也會有譯文。代價是切換要等幾秒（收尾加上新模型載入，8B 較久），這段時間的系統音訊不會被收錄，錄音檔也不會有；要避開重要發言就在空檔切換。暫停或已終止時，下次開始或繼續才套用。

每句送給 PLaMo 時會帶前兩句原文當前文，以及術語表裡的詞（Latin 字母的詞一律帶；含漢字的日文詞只在譯成繁體中文時帶；含假名的詞不帶，避免譯文照抄假名）。術語表不是絕對強制，模型偶爾仍會改寫。譯成英文或繁體中文的結果含有假名時，worker 會不帶前文與術語表重翻一次。

`Logs/app.log` 每句記一行 `translate utterance=… 秒數 ja->zh-hant 原文 -> 譯文`。翻譯模型載入失敗時逐字稿照常運作，譯文按鈕標題變成「譯文：載入失敗」，原因放在按鈕的提示文字，也寫在 `Logs/app.log`。單句翻譯失敗的警告也顯示在提示文字裡，下一句翻成功就清掉。

翻譯跟不上時（例如連續講話）：定稿優先，Qwen 重解在處理時翻譯會等；翻譯佇列超過 3 句就丟掉最舊的，被丟的句子只有原文、沒有譯文。

舊版的 `Models/polish-model.txt`（重解開關）App 啟動時自動遷移：已下載 Qwen3-ASR 且沒有 `final-model.txt` 就寫入 `final-model.txt`（重解開啟），然後刪掉 `polish-model.txt`。舊版留下的 `Models/speech-engine.txt` 與舊模型目錄（parakeet、Hy-MT2 等）程式不會讀，可以自行刪除。

### 術語表與刪贅詞（預設開啟）

每句定稿後、翻譯前，worker 先做兩件不經過模型的處理，結果直接替換字幕與逐字稿上的原文，譯文也以處理後的文字為準：

1. 規則刪贅詞：刪掉 えっと、えーと、えー、あのー、うーん、句首接假名或頓號的 あの（「あの件」不碰）、英文的 um／uh，合併「we can, we can」這種兩個字以上的重講。「なんか」「まあ」有語意，不碰。
2. 術語表替換：`Models/glossary.txt`，一行一條，自己維護（Models/ 不進 Git）：

```text
ショピファイト=Shopify      # 翻譯前直接把左邊換成右邊；英文不分大小寫、以字為單位，片假名直接比對
sopifi=Shopify
line=LINE                  # 只差大小寫的對照只套在日文句子上；英文句子裡的 line 多半是普通單字
Figma                      # 單獨一行：不替換，只給翻譯提示與 Apple 辨識參考
```

術語表的正確寫法會一併當作翻譯的 glossary，也會當作 SpeechAnalyzer 的 contextual strings，讓辨識在第一次就偏向這些詞。改完存檔，下次開始聆聽時生效（終止後再開始，或切換譯文／模型；暫停再繼續不會重讀）。

worker 啟動時在 `Logs/app.log` 記 `glossary: N replacements, M terms`，App 另記 `glossary: M contextual strings`。

### Qwen 重解（預設開啟）

每句定稿時把該句音訊交給 Qwen3-ASR-1.7B（8-bit）重解一次，結果取代 SpeechAnalyzer 的文字。

worker 只載入 Qwen3-ASR（峰值約 3.6 GB）和 PLaMo。依 App 判定的語言指定 Qwen 語言（日文用 Japanese、英文用 English）：自動判斷語言時，日英夾雜的音訊會被翻成英文；英文句用日文模式解會變假名或無限重複。重解結果若有重複迴圈、比原文長太多、英文句含假名、日文句沒有日文字，或沒有任何句子對得上原文，就丟棄沿用原文（log 標 `rejected: 原因`）。語言以 App 的判定為準，不因重解結果改變。1.5 秒以下的日文短句沿用 SpeechAnalyzer 的文字。每句約多等 1～2.5 秒。和翻譯可以同時開，16 GB 機器上 Qwen3-ASR 加 PLaMo 2B 合計約 5 GB。

`Scripts/download_models.py` 會下載模型；舊環境要自己補：

```bash
cd "${REPO_ROOT}"
uv pip install --python .venv/bin/python "mlx-audio[stt]"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-ASR-1.7B-8bit', local_dir='Models/Qwen3-ASR-1.7B-8bit')"
```

點底列的「Qwen 重解：關／開」切換。正在聽時先收尾再切換（同上），暫停時下次繼續才套用。載入 Qwen3-ASR 失敗時，worker 會降級成不重解（沿用 SpeechAnalyzer 的文字）繼續翻譯。設定寫在 `Models/final-model.txt`（內容為模型目錄名；檔案不存在就是關閉）。`Logs/app.log` 每句記一行 `refine utterance=… 語言 秒數 audio=… 原文 -> 重解結果`，可以拿來對照。

### 翻譯前潤稿（選用，預設關閉）

在定稿和翻譯之間多一個 LLM 步驟（Qwen3-4B 4-bit），把原文整理乾淨後再翻譯：刪掉規則抓不到的贅詞與重講、補標點、把聽錯的專有名詞改成 `Models/glossary.txt` 裡的寫法。輸出仍是原文的語言，不翻譯、不摘要、不改意思。驗證會保證數字序列完全相同、否定詞（ない、ません、じゃ、ず、not、n't、no 等）不會變少；人名不在這兩項保證內，只靠下面「只准刪贅詞與重講、替換限術語」的限制間接約束（人名不會被換成別的詞，但若剛好是重講會被合併）。

- 逐字稿不會因潤稿而延遲：定稿的原文先照常顯示，潤稿有改動時再換成潤稿後的文字，翻譯以潤稿後的文字為準。
- 潤稿結果要通過驗證才採用：英文句不能出現日文字、不能縮太多、術語表裡的詞不能被改掉、數字與否定不能變；改動只准是刪贅詞（えっと、あの、um、you know 等整詞）、刪緊鄰的完整重講（そんなそんな、we we；疊字詞如いろいろ、まだまだ不算重講）、加標點，以及把聽錯的片假名／英文字換成術語表的寫法（被換掉的原文要跟術語讀音或拼法相近，不能含數字與否定）；不過就沿用原文。生成逾時（3 秒起，依句子長度放寬，上限 6 秒）、出錯或被拒絕時也沿用原文。日文少於 7 字、英文少於 4 個詞的句子不潤稿。
- 代價：每句多約 1～3 秒才開始翻譯（M1 Air 16 GB 實測，PLaMo 與 Qwen3-ASR 同時常駐），多佔約 2.4 GB 記憶體，三個模型合計約 7.3 GB。
- 較小的 Qwen3-1.7B 試過：要嘛幾乎不改，要嘛把前文或示例抄進輸出，不採用。

下載（約 2.3 GB，已下載的模型會略過）：

```bash
cd "${REPO_ROOT}"
LIVE_TRANSLATOR_POLISH=1 HF_HUB_DISABLE_XET=1 .venv/bin/python Scripts/download_models.py
```

點底列的「潤稿：關／開」切換；沒下載模型時會跳出提示。正在聽時先收尾再切換，暫停時下次繼續才套用。設定寫在 `Models/prepolish-model.txt`（內容為模型目錄名；檔案不存在、內容為空或為 `none`，或 `Models/Qwen3-4B-4bit/config.json` 不存在都視為關閉）。載入潤稿模型失敗時，worker 會降級成不潤稿繼續翻譯，按鈕顯示「潤稿：載入失敗」、原因在 tooltip，重新開始聆聽後恢復。翻譯最多等潤稿 2 秒，超過就用目前的文字照樣翻譯。

看 `Logs/app.log`：採用的句子記成 `polish utterance=3 ja 1.25s 原文 -> 潤稿後`（`ja` 或 `en` 是語言欄位）；沒有採用的記成 `polish utterance=3 ja 1.25s 原文 -> 模型輸出 rejected: 原因`。模型沒有改動時原因是 `same`；其他原因有 language、particle、length、`term 詞`、number、negation、empty、timeout、truncated，以及逐段比對不過時的 `原因: 原文片段→潤稿片段`（原因有 insert、source、not term、dissimilar、number、short、reduplicated、not repeat）。

## 10. 問題排除

| 症狀 | 原因 | 處理 |
|---|---|---|
| 狀態列「尚未完成安裝」 | 找不到 `.venv/bin/python` 或 worker（模型不在此檢查；沒有 PLaMo 時只出逐字稿，按鈕顯示「譯文：未安裝」；沒有 Qwen3-ASR 時按下「Qwen 重解」會提示缺少模型） | 回到第 2 節；App 是否還在 `Build/` 裡 |
| 狀態列「無法啟動語音辨識」 | SpeechAnalyzer 語言資產下載失敗，通常是第一次啟動時沒有網路 | 連上網路後重開 App |
| 一直「聆聽中」但沒有字幕 | 沒有錄製權限，或重編後簽章變了、授權失效 | 第 7 節重新授權；建立固定簽章（第 6 節） |
| 字幕延遲明顯變長、整台機器變慢 | 記憶體不足開始 swap | 關掉瀏覽器等大程式；活動監視器看「記憶體壓力」與「已使用的交換空間」 |
| 狀態列「處理時發生錯誤」 | worker 例外 | 看 `Logs/app.log` 的最後幾行 |
| 有逐字稿但沒有譯文 | 沒下載 PLaMo（按鈕顯示「譯文：未安裝」），或翻譯模型載入失敗 | 第 4 節下載；看 `Logs/app.log` 的 `translator load failed` |
| 某場會議錯特別多 | 系統負載造成掉幀（最常見），或少數情況下音訊本身窄頻、講者麥克風悶 | `Scripts/audio_qc.py` 看斷音與頻寬判讀（第 8 節）；斷音多就關掉瀏覽器等大程式、不要在會議中切模型；專有名詞聽錯則補術語表 |

`Logs/app.log` 記錄每次啟動、狀態切換、按鈕操作與 worker 的錯誤輸出。要看 worker 的解碼細節，帶環境變數啟動：

```bash
open --env LIVE_TRANSLATOR_DEBUG=1 "${REPO_ROOT}/Build/LiveTranslator.app"
```

細節同樣寫進 `Logs/app.log`。
