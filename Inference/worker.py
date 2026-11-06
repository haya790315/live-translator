import base64
import difflib
import json
import os
import queue
import re
import sys
import threading
import time
import unicodedata

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 載入 PLaMo 時 transformers 會印一行 model type 警告，無害，壓掉
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler


SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 4
CONTEXT_UTTERANCES = 2
MAX_TRANSLATION_TOKENS = 160
MAX_REFINE_TOKENS = 200
# 重解的 token 上限依音訊長度算（每秒 token 數 + 常數），避免卡在重複迴圈跑滿上限
REFINE_TOKENS_PER_SECOND = {"ja": 8, "en": 5}
REFINE_TOKEN_BASE = 16
REFINE_REPETITION_PENALTY = 1.05
REFINE_LANGUAGE = {"ja": "Japanese", "en": "English"}
REFINE_MATCH = 0.5
# 重解結果比原文長太多就視為失控（正規化後字數 > 原文 * 倍數 + 常數）
REFINE_MAX_GROWTH = 1.8
REFINE_MAX_EXTRA_CHARS = 10
HISTORY_LIMIT = 50
MAX_PENDING_TRANSLATIONS = 3
# 翻譯最多等定稿／潤稿這麼久（秒）：潤稿佇列塞住時不能讓譯文無限期延後
TRANSLATE_WAIT_MAX = 2.0
PLAMO_EOS = {1, 16}
PLAMO_TAG = re.compile(r"<\|plamo:[^|>]*\|>")
PLAMO_MARK = "<|plamo:"
TARGETS = ("zh-hant", "ja-en")
# 翻譯前潤稿：太短的句子不潤稿；生成超過逾時就放棄、用原文；結果比原文短太多（正規化後）視為刪過頭
POLISH_MIN_JA_CHARS = 7
POLISH_MIN_EN_WORDS = 4
# 逾時依句子長度放寬：基本 3 秒，每 40 字（英文每 40 字元）多 1 秒，上限 6 秒（4B 在 M1 上每秒約 25 字）
POLISH_TIMEOUT = 3.0
POLISH_TIMEOUT_PER_CHARS = 40
POLISH_TIMEOUT_MAX = 6.0
MAX_POLISH_TOKENS = 200
POLISH_MIN_LENGTH = 0.6
PREFIX_PROBE = "テスト"

LATIN = re.compile(r"[A-Za-z]")
JAPANESE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
DEBUG = os.environ.get("LIVE_TRANSLATOR_DEBUG") == "1"
HALLUCINATIONS = {"ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録お願いします", "チャンネル登録をお願いします", "字幕視聴ありがとうございました"}
FILLERS = {"えっと", "えーと", "ええと", "えー", "え", "あー", "あ", "あの", "あのー", "うーん", "うん", "ん", "んー", "んん", "ピッ", "はあ", "ふー", "まあ", "um", "uh", "uhm", "umm", "hmm", "hm", "mm", "mhm", "er", "ah", "oh", "huh"}
FILLER_SPLIT = re.compile(r"[、。！？!?…ー\s.,]+")
NORMALIZE = re.compile(r"[\s。、！？!?…]")
MATCH_STRIP = re.compile(r"[\s。、，,.！？!?…「」『』\"'’]")
REFINE_SPLIT = {"ja": re.compile(r"(?<=[。！？!?])"), "en": re.compile(r"(?<=[.?!])\s+")}
CJK_SPACE = re.compile(r"(?<=[^\x00-\x7f])[ \u3000]+(?=[^\x00-\x7f])|^[ \u3000]+|[ \u3000]+$")
# 嚴格的假名範圍：不含 ・(U+30FB) ー(U+30FC)，「唐納・川普」不算假名
KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa]")

# 規則刪贅詞：只刪沒有歧義的語氣詞，不經過模型。「あの」只刪句首且後面接假名或頓號的（「あの件」不碰）；
# 「なんか」「まあ」有語意，不碰。英文另外合併「we can, we can」這種兩個字以上的重講
JA_FILLER = re.compile(r"(?:えーっと|えっとー|えっと|えーと|ええと|えーっ|えー|あのー|あのう|うーん|んー)[、，]?")
JA_LEADING_ANO = re.compile(r"^あの(?=[、，぀-ゟ])")
EN_FILLER = re.compile(r"\b(?:um+|uh+|uhm+|erm?|hmm+)\b", re.IGNORECASE)
EN_REPEAT = re.compile(r"\b((\w+)(?:\s+\w+){1,3}),?\s+\1\b", re.IGNORECASE)
MEANINGFUL = re.compile(r"[぀-ヿ一-鿿A-Za-z0-9]")
LATIN_WORD = re.compile(r"[A-Za-z]+")
# 刪贅詞後殘留的助詞開頭（えーとあの僕 → と僕）
PARTICLE_START = re.compile(r"^[とがをにはも](?=[^\x00-\x7f])")

emit_lock = threading.Lock()
gpu_lock = threading.Lock()


def emit(kind, utterance=0, text="", translation="", message="", **extra):
    payload = {"kind": kind, "utterance": utterance, "text": text, "translation": translation, "message": message}
    payload.update(extra)
    with emit_lock:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def debug(message):
    if DEBUG:
        sys.stderr.write(f"[worker {time.monotonic():.2f}] {message}\n")
        sys.stderr.flush()


# 不受 DEBUG 控制，App 會把 stderr 接到 Logs/app.log
def note(message):
    sys.stderr.write(f"[worker] {message}\n")
    sys.stderr.flush()


def is_filler(text):
    pieces = [piece.lower() for piece in FILLER_SPLIT.split(text) if piece]
    return bool(pieces) and all(piece in FILLERS for piece in pieces)


# 有假名就是日文（日英夾雜的「LINEのbotをdeployしました」算日文）；沒有假名才看拉丁字母；純漢字算日文
def language_of(text):
    if KANA.search(text):
        return "ja"
    return "en" if LATIN.search(text) else "ja"


class Refiner:
    def __init__(self, path):
        from mlx_audio.stt.utils import load as load_stt
        self.model = load_stt(path)
        self.model.generate(np.zeros(SAMPLE_RATE, dtype=np.float32), max_tokens=8)

    # 語言以 App 判定的為準：日文一律指定 Japanese（自動判斷時日英夾雜的音訊會被翻成英文、日文句子整句消失），
    # 英文指定 English（用日文模式解英文會變假名或無限重複）
    def transcribe(self, samples, language):
        seconds = len(samples) / SAMPLE_RATE
        max_tokens = min(MAX_REFINE_TOKENS, int(seconds * REFINE_TOKENS_PER_SECOND[language]) + REFINE_TOKEN_BASE)
        with gpu_lock:
            output = self.model.generate(samples, max_tokens=max_tokens, language=REFINE_LANGUAGE[language], repetition_penalty=REFINE_REPETITION_PENALTY)
        text = output.text.strip()
        return CJK_SPACE.sub("", text) if language == "ja" and JAPANESE.search(text) else text


def overlap(sentence, original):
    a = MATCH_STRIP.sub("", sentence).lower()
    b = MATCH_STRIP.sub("", original).lower()
    if not a or not b:
        return 0.0
    matched = sum(block.size for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if block.size >= 2)
    return matched / len(a)


# 同一個 1～4 字（英文為 1～4 個詞）的片段連續出現 4 次以上
CHAR_LOOP = re.compile(r"(.{1,4}?)\1{3,}")
WORD_LOOP = re.compile(r"\b((?:\w+ ){0,3}\w+)(?: \1\b){3,}", re.IGNORECASE)


def has_repetition(text, language):
    if language == "en":
        words = re.findall(r"[\w']+", text.lower())
        if WORD_LOOP.search(" ".join(words)):
            return True
        return len(words) >= 8 and len(set(words)) / len(words) < 0.4
    chars = MATCH_STRIP.sub("", text)
    if CHAR_LOOP.search(chars):
        return True
    return len(chars) >= 12 and len(set(chars)) / len(chars) < 0.25


# 重解結果不可信的原因；可信回傳 None
def refine_reject_reason(original, refined, language):
    if not refined:
        return "empty"
    if NORMALIZE.sub("", refined) in HALLUCINATIONS or is_filler(refined):
        return "filler"
    if has_repetition(refined, language):
        return "repeat"
    if len(MATCH_STRIP.sub("", refined)) > len(MATCH_STRIP.sub("", original)) * REFINE_MAX_GROWTH + REFINE_MAX_EXTRA_CHARS:
        return "too long"
    if language == "en" and KANA.search(refined):
        return "kana in english"
    if language == "ja" and not JAPANESE.search(refined):
        return "no japanese"
    return None


# 沒有任何句子對得上原文時回傳原文，不信任重解
def choose_final(original, refined, language):
    if language not in REFINE_SPLIT or refine_reject_reason(original, refined, language):
        return original
    sentences = [piece.strip() for piece in REFINE_SPLIT[language].split(refined) if piece.strip()]
    kept = [index for index, sentence in enumerate(sentences) if overlap(sentence, original) >= REFINE_MATCH]
    if not kept:
        return original
    return (" " if language == "en" else "").join(sentences[kept[0]: kept[-1] + 1])


def remove_fillers(text, language):
    if language == "en":
        cleaned = EN_FILLER.sub("", text)
        cleaned = EN_REPEAT.sub(r"\1", cleaned)
        cleaned = re.sub(r"\s*,(?:\s*,)+", ",", cleaned)
        cleaned = re.sub(r"\s+([,.?!])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
        if cleaned and text[:1].isupper() and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
    else:
        cleaned = JA_FILLER.sub("", text)
        cleaned = JA_LEADING_ANO.sub("", cleaned)
        cleaned = re.sub(r"[、，]{2,}", "、", cleaned).lstrip("、，")
    return cleaned if MEANINGFUL.search(cleaned) else text


class Glossary:
    """Models/glossary.txt 的靜態條目（使用者維護）：聽錯寫法 → 正確寫法，套用時不經過模型。"""

    def __init__(self):
        self.entries = {}
        self.extra = []

    # 檔案格式：一行一條，「聽錯寫法=正確寫法」會替換；只寫「正確寫法」則只給翻譯與辨識參考；# 開頭是註解
    def load(self, path):
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    wrong, right = (part.strip() for part in line.split("=", 1))
                    if wrong and right and wrong != right:
                        self.entries[wrong] = right
                elif line not in self.extra:
                    self.extra.append(line)

    # 只差大小寫的對照（line → LINE）只套在日文句子上；英文句子裡的 line 多半是普通單字
    def apply(self, text, language="ja"):
        # 先套英文對照：片假名換成英文後，緊鄰的英文字（ショピファイトline）會失去字界
        for wrong in sorted(self.entries, key=lambda item: (not LATIN.search(item), -len(item))):
            right = self.entries[wrong]
            if LATIN.search(wrong):
                if language == "en" and wrong.lower() == right.lower():
                    continue
                text = re.sub(rf"(?<![A-Za-z]){re.escape(wrong)}(?![A-Za-z])", right, text, flags=re.IGNORECASE)
            else:
                # 片假名換成英文後若緊鄰另一個英文字（ショピファイトLINE），補空白，翻譯才不會把兩個名字黏成一個
                def replace(match, right=right):
                    before = match.string[match.start() - 1] if match.start() > 0 else ""
                    after = match.string[match.end()] if match.end() < len(match.string) else ""
                    lead = " " if LATIN.match(before) and LATIN.match(right) else ""
                    trail = " " if LATIN.match(after) and LATIN.search(right[-1:]) else ""
                    return lead + right + trail
                text = re.sub(re.escape(wrong), replace, text)
        return text

    def terms(self):
        return list(dict.fromkeys(self.extra + list(self.entries.values())))


# 術語表檔案：環境變數優先，否則 Models/glossary.txt；不存在就沒有靜態術語
def glossary_path(models_dir):
    path = os.environ.get("LIVE_TRANSLATOR_GLOSSARY") or os.path.join(models_dir, "glossary.txt")
    return path if os.path.isfile(path) else None



class Translator:
    """PLaMo 3 Translate：背景執行緒依序翻譯定稿的句子，前 CONTEXT_UTTERANCES 句原文當 context，術語表當 glossary。
    mode 是 zh-hant（日、英都翻成繁體中文）或 ja-en（日→英、英→日）。"""

    def __init__(self, path, mode, glossary):
        self.mode = mode
        self.glossary = glossary
        self.model, self.tokenizer = load(path, tokenizer_config={"trust_remote_code": True})
        self.raw = self.tokenizer._tokenizer
        self.sampler = make_sampler(temp=0.0)
        self.eos = set(PLAMO_EOS) | set(getattr(self.tokenizer, "eos_token_ids", None) or ())
        self.history = []
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.jobs = []
        self.closed = False
        # 定稿優先：Finalizer 有待處理工作時不開始新的翻譯（由 Finalizer 建好後設定）
        self.finalizer = None
        # 先翻一句暖機：第一次推論要編譯，不先做的話第一句字幕會慢好幾秒
        try:
            self._generate("こんにちは。", "ja", self._target("ja"))
        except Exception as exc:
            note(f"translator warm-up failed: {exc}")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _target(self, language):
        if self.mode == "ja-en":
            return "en" if language == "ja" else "ja"
        return "zh-hant"

    def submit(self, utterance, text, language):
        with self.lock:
            # 同一句被修正後重送，舊的還沒翻就丟掉
            self.jobs = [job for job in self.jobs if job[0] != utterance]
            self.jobs.append((utterance, text, language))
            # 翻譯跟不上時丟最舊的（不翻，但仍留在前文裡）
            while len(self.jobs) > MAX_PENDING_TRANSLATIONS:
                dropped = self.jobs.pop(0)
                self._remember(dropped[0], dropped[1])
                note(f"translation queue over {MAX_PENDING_TRANSLATIONS}, dropped utterance={dropped[0]} {dropped[1]!r}")
            self.wake.notify()

    # 呼叫時須持有 self.lock
    def _remember(self, utterance, text):
        self.history = [item for item in self.history if item[0] != utterance]
        self.history = sorted(self.history + [(utterance, text)])[-HISTORY_LIMIT:]

    # 定稿優先：等到 Finalizer 空閒才開始新的生成（已開始的生成照跑完），但最多等 TRANSLATE_WAIT_MAX 秒，超過就照樣翻
    def _wait_for_finalizer(self):
        deadline = time.monotonic() + TRANSLATE_WAIT_MAX
        while self.finalizer is not None and self.finalizer.busy():
            with self.lock:
                if self.closed and not self.jobs:
                    return
            if time.monotonic() >= deadline:
                debug(f"finalizer still busy after {TRANSLATE_WAIT_MAX}s, translating anyway")
                return
            time.sleep(0.05)

    def close(self):
        with self.lock:
            self.closed = True
            self.wake.notify()
        self.thread.join()

    def _context(self, utterance):
        texts = [text for number, text in self.history if number < utterance][-CONTEXT_UTTERANCES:]
        context = " ".join(texts)
        return context if PLAMO_MARK not in context else ""

    # Latin 字母的詞一律帶；含日文的詞只在譯成繁體中文時帶，避免把日文詞塞進英文譯文。
    # 含假名的詞連繁體中文也不帶：譯文照抄假名會觸發「含假名就重翻」
    def _glossary(self, target):
        items = []
        for term in self.glossary.terms():
            if PLAMO_MARK in term or (JAPANESE.search(term) and target != "zh-hant") or (target == "zh-hant" and KANA.search(term)):
                continue
            items.append({"source": term, "target": term})
        return items

    def _generate(self, text, source, target, context="", glossary=()):
        options = {"target_lang": target, "source_lang": source}
        if context:
            options["context"] = context
        if glossary:
            options["glossary"] = list(glossary)
        prompt = self.raw.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, **options)
        tokens = self.raw.encode(prompt, add_special_tokens=False)
        pieces = []
        with gpu_lock:
            for response in stream_generate(self.model, self.tokenizer, prompt=tokens, max_tokens=MAX_TRANSLATION_TOKENS, sampler=self.sampler):
                if response.token in self.eos:
                    break
                pieces.append(response.text)
        return PLAMO_TAG.sub("", "".join(pieces)).strip()

    def _translate(self, text, language, target, context):
        translation = self._generate(text, language, target, context, self._glossary(target))
        if target != "ja" and KANA.search(translation):
            debug(f"kana in translation {translation!r}, retrying without context")
            translation = self._generate(text, language, target) or translation
        return translation

    def _run(self):
        while True:
            with self.lock:
                while not self.jobs and not self.closed:
                    self.wake.wait()
                if not self.jobs and self.closed:
                    return
                utterance, text, language = self.jobs[0]
            self._wait_for_finalizer()
            with self.lock:
                # 等待期間若這句已被丟掉或換成修正版，重新取
                if not self.jobs:
                    continue
                utterance, text, language = self.jobs.pop(0)
                context = self._context(utterance)
                self._remember(utterance, text)
            target = self._target(language)
            started = time.monotonic()
            try:
                translation = self._translate(text, language, target, context)
            except Exception as exc:
                note(f"translate failed utterance={utterance}: {exc}")
                emit("warning", utterance, message=f"翻譯失敗：{exc}", code="translate_failed")
                continue
            note(f"translate utterance={utterance} {time.monotonic() - started:.2f}s {language}->{target} {text!r} -> {translation!r}")
            emit("final", utterance, text, translation=translation)


# SpeechAnalyzer 在很短的日文句子上通常是對的，時間戳卻常偏移，重解反而會吃進前後句；這種句子沿用原文。
# 判成英文的短句仍要重解：多半是日文被誤判成英文的亂碼（It's my feet.）
MIN_REFINE_SECONDS = 1.5
# App 切片時前後各多留的秒數（LiveTranslator.swift 的 refineLeadSeconds + refineTailSeconds）
REFINE_PADDING_SECONDS = 0.2

# 重解的切片會混進上一句的尾巴：新結果開頭與上一句結尾重複 MIN_REPEAT_CHARS 字以上就切掉（比對時忽略標點與空白）
MIN_REPEAT_CHARS = 4


def strip_repeated_head(text, previous):
    head = MATCH_STRIP.sub("", text).lower()
    tail = MATCH_STRIP.sub("", previous).lower()
    size = next((k for k in range(min(len(head), len(tail)), MIN_REPEAT_CHARS - 1, -1) if tail.endswith(head[:k])), 0)
    if size == 0 or size == len(head):
        return text
    seen = 0
    for index, char in enumerate(text):
        if not MATCH_STRIP.match(char):
            seen += 1
            if seen == size:
                return text[index + 1:].lstrip("。、，,.！？!? ")
    return text


POLISH_SYSTEM_PROMPTS = {
    "ja": """あなたは音声認識の書き起こしを整える校正者です。【本文】だけを整え、整えた日本語の文だけを出力します。【前文】は文脈を理解するための参考で、出力には含めません。
ルール:
- 削るのは、言い淀み（えー、えっと、えと、あの、あのー、んー）と、言い直しや言い間違いで繰り返された部分（そんなそんな → そんな）だけ。はい、うん、じゃあ、まあ、そう などの相づちやつなぎ言葉は残す
- 句読点を補う
- 聞き間違いと思われる固有名詞は、【用語】の正しい表記に直す
- 数字・否定・人名は変えない。意味を変えない。言葉を足さない。要約しない。翻訳しない。言い換えない
- 語尾や口調は変えない（話し言葉のまま）
- 直すところがなければ元の文をそのまま出力する
- 出力は整えた文だけ（説明・引用符・前置きは書かない）
【用語】{terms}

例
【前文】なし
【本文】サーバーの移行はあのサーバーの移行は来週の金曜日までに終わる予定ですえっと確認お願いします
サーバーの移行は来週の金曜日までに終わる予定です。確認お願いします。

【前文】次はショッピファイの件ですね。
【本文】ショッピファイの決済まわりでちょっと確認したいことがありますあのそんなそんな大きな話じゃないです
Shopifyの決済まわりでちょっと確認したいことがあります。そんな大きな話じゃないです。

【前文】なし
【本文】じゃあまあ明日は十五時で行きましょう
じゃあまあ、明日は十五時で行きましょう。""",
    "en": """You clean up speech-recognition transcripts. Fix only the [Sentence] and output only the cleaned English sentence. [Context] is for understanding only; never include it in the output.
Rules:
- Remove only filler words (um, uh, er, you know) and false starts or repeated words (we we should -> we should)
- Add punctuation and capitalization
- Fix proper nouns that were clearly misheard, using the spelling in [Terms]
- Never change numbers, negations or names. Keep the meaning. Add nothing. Do not summarize. Do not translate. Do not rephrase
- Keep the spoken tone
- If there is nothing to fix, output the sentence unchanged
- Output the cleaned sentence only, with no explanation, quotes or preface
[Terms] {terms}

Examples
[Context] none
[Sentence] so we we should probably sync with the the backend team first
So we should probably sync with the backend team first.

[Context] none
[Sentence] i think the sopifi integration is is not ready yet
I think the Shopify integration is not ready yet.""",
}
POLISH_LABELS = {"ja": ("【前文】", "【本文】", "なし"), "en": ("[Context] ", "[Sentence] ", "none")}


# 潤稿只准刪的口語贅詞（整詞比對）。日文單字元的 え/あ/ん 不在這裡，見 JA_SINGLE_FILLER；
# 「うん」「まあ」「はい」是相づち、「like」有語意，都不放行
POLISH_FILLERS = {
    "ja": ["えっと", "えーと", "ええと", "えーっと", "えっとー", "えと", "えー", "あー", "あのー", "あのう", "あの", "うーん", "んー"],
    "en": ["um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "you know", "i mean"],
}
# 單字元的 え/あ/ん 只有前後是標點、空白或句首句尾時才算贅詞（いいえ、あいう 裡的不算）
JA_SINGLE_FILLER = re.compile(r"(?:^|(?<=[、。！？!?…，,.\s]))[えあん](?=[、。！？!?…，,.\s]|$)")
# 疊字詞：重複是詞本身的一部分，刪掉一半會變成另一個詞
JA_REDUPLICATED = {"いろいろ", "まだまだ", "だんだん", "どんどん", "ますます", "それぞれ", "ときどき", "そろそろ", "ちょこちょこ", "もっともっと", "どきどき", "わくわく", "ぎりぎり", "ばらばら", "ぼろぼろ", "さらさら", "ふらふら", "ゆらゆら", "くるくる", "ころころ", "ぐるぐる", "ぺらぺら"}
NEGATION = {"ja": re.compile(r"ない|ません|じゃ|ず"), "en": re.compile(r"\b(?:not|never|no|cannot)\b|n't")}
EN_NEGATION_WORDS = {"not", "never", "no", "cannot", "nor", "without"}
DIGIT_RUN = re.compile(r"[0-9〇一二三四五六七八九十百千万億]+")
DIGIT_CHAR = re.compile(r"[0-9〇一二三四五六七八九十百千万億]")
KATAKANA_LATIN = re.compile(r"[ァ-ヺーa-z]+")
# 被換掉的原文段與術語的相似度下限（片假名先轉羅馬字再比）
POLISH_REPLACE_SIMILARITY = 0.4
# 潤稿後 ASCII 詞與日文之間若被補了空白（一旦 Shopifyの）就拿掉；日文本來就不在這裡留空白
JA_ASCII_SPACE = re.compile(r"(?<=[^\x00-\x7f]) +(?=[A-Za-z0-9])|(?<=[A-Za-z0-9]) +(?=[^\x00-\x7f])")
SENTENCE_START = re.compile(r"(?:^|[.!?。！？]\s*)$")

_ROMAJI = dict(zip(
    "アイウエオカキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデドナニヌネノハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヲンヴ",
    "a i u e o ka ki ku ke ko ga gi gu ge go sa shi su se so za ji zu ze zo ta chi tsu te to da ji zu de do na ni nu ne no ha hi fu he ho ba bi bu be bo pa pi pu pe po ma mi mu me mo ya yu yo ra ri ru re ro wa o n vu".split(),
))
_SMALL_VOWEL = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
# 比對用的粗略正規化：l/r、v/b、f/h 不分
_ROMAJI_FOLD = str.maketrans({"l": "r", "v": "b", "f": "h"})


# 片假名轉粗略羅馬字（只為了跟拉丁字母的術語比相似度）；拉丁字母原樣小寫
def romanize(text):
    out = []
    double = False
    for char in text.lower():
        if char == "ッ":
            double = True
        elif char == "ー":
            continue
        elif char in _SMALL_VOWEL and out:
            out[-1] = out[-1][:-1] + _SMALL_VOWEL[char]
        elif char in _SMALL_Y and out:
            base = out[-1]
            out[-1] = (base[:-1] if base.endswith("i") else base)
            out[-1] += _SMALL_Y[char] if out[-1].endswith(("sh", "ch", "j")) else "y" + _SMALL_Y[char]
        else:
            syllable = _ROMAJI.get(char, char)
            if double and syllable[:1].isalpha():
                syllable = syllable[0] + syllable
            double = False
            out.append(syllable)
    return "".join(out).translate(_ROMAJI_FOLD)


def is_polishable(text, language):
    if language == "en":
        return len(text.split()) >= POLISH_MIN_EN_WORDS
    return len(MATCH_STRIP.sub("", text)) >= POLISH_MIN_JA_CHARS


# 比對用的詞：日文是逐字（連續的拉丁字母／數字算一個詞），英文是以空白與標點切開的字，一律小寫
def polish_tokens(text, language):
    text = text.lower().replace("’", "'")
    if language == "en":
        return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", MATCH_STRIP.sub("", text))


def filler_tokens(language):
    return sorted((tuple(polish_tokens(word, language)) for word in POLISH_FILLERS[language]), key=len, reverse=True)


def digit_runs(text, keys):
    text = unicodedata.normalize("NFKC", MATCH_STRIP.sub("", text)).lower()
    for key in keys:
        text = text.replace(key, "")
    return DIGIT_RUN.findall(text)


# 英文大小寫還原：只有術語表的字、句首字母大寫、代名詞 I 放行；其餘改回原文的寫法（sopifi → SOPIFI 會被翻譯當成縮寫照抄）。
# 同一個字在原文有多種寫法（Apple and apple）就不動潤稿的輸出，不猜對應哪一個
def restore_case(original, polished, terms):
    forms = {}
    for word in LATIN_WORD.findall(original):
        forms.setdefault(word.lower(), set()).add(word)
    term_words = {word for term in terms for word in LATIN_WORD.findall(term)}

    def fix(match):
        word = match.group()
        known = forms.get(word.lower())
        if not known or len(known) > 1 or word in known or word in term_words or word == "I":
            return word
        original_form = next(iter(known))
        if SENTENCE_START.search(match.string[: match.start()]) and word == original_form[:1].upper() + original_form[1:]:
            return word
        return original_form

    return LATIN_WORD.sub(fix, polished)


# 刪除段：先去掉兩端的贅詞，剩下的只有「緊鄰的完整重講」或「重複了上一句結尾」才放行
def deletion_reason(tokens, start, end, i1, i2, previous, language):
    segment = tokens[start:end]
    if not segment:
        return None
    unit = "".join(segment)
    if DIGIT_CHAR.search(unit):
        return "number"
    size = len(segment)
    if language == "ja":
        if len(unit) < 2:
            return "short"
        if (len(unit) <= 2 and re.fullmatch(r"[぀-ヿ]+", unit)) or unit + unit in JA_REDUPLICATED:
            return "reduplicated"
    adjacent = tokens[i2:i2 + size] == segment or (i1 >= size and tokens[i1 - size:i1] == segment)
    if adjacent:
        return None
    tail_min = 4 if language == "ja" else 2
    if i1 == 0 and len(unit) >= tail_min and len(previous) >= size and previous[-size:] == segment and (language == "ja" or size >= 2):
        return None
    return "not repeat"


# 替換段：只准把聽錯的專有名詞改成術語。被換掉的原文必須是片假名／拉丁字母（日文）或 1～3 個英文詞，
# 不含數字與否定、不比術語長太多，而且讀音／拼法要跟術語相近
def replacement_reason(deleted, inserted, language, keys):
    d = "".join(deleted)
    s = "".join(inserted)
    if DIGIT_CHAR.search(d):
        return "number"
    if language == "ja":
        if not KATAKANA_LATIN.fullmatch(d):
            return "source"
    elif not 1 <= len(deleted) <= 3 or any(word in EN_NEGATION_WORDS or "n't" in word for word in deleted) or not re.fullmatch(r"[a-z' ]+", " ".join(deleted)):
        return "source"
    candidates = [key for key in keys if s in key]
    if not candidates:
        return "not term"
    folded = romanize(d)
    for key in candidates:
        if len(d) <= len(key) * 2 + 2 and difflib.SequenceMatcher(None, folded, romanize(key)).ratio() >= POLISH_REPLACE_SIMILARITY:
            return None
    return "dissimilar"


def accept_polish(original, polished, language, terms=(), previous=""):
    """潤稿只准刪贅詞、刪重講、加標點，以及把聽錯的字改成術語表裡的寫法；其他改動一律判定不過、沿用原文。
    另外整句檢查：數字序列必須相同、否定詞不能變少。回傳 (採用的文字或 None, 原因)。"""
    if not polished or polished == original:
        return None, "same"
    # 日文輸出不查假名：翻成別的語言會在下面的逐段比對被擋下
    if language == "en" and JAPANESE.search(polished):
        return None, "language"
    if language == "ja" and PARTICLE_START.match(polished) and not original.startswith(polished[0]):
        return None, "particle"
    polished = restore_case(original, polished, terms)
    cleaned = JA_SINGLE_FILLER.sub("", original) if language == "ja" else original
    checked = JA_SINGLE_FILLER.sub("", polished) if language == "ja" else polished
    a_tokens = polish_tokens(cleaned, language)
    b_tokens = polish_tokens(checked, language)
    sep = " " if language == "en" else ""
    a, b = sep.join(a_tokens), sep.join(b_tokens)
    if not a or not b:
        return None, "empty"
    if a == b:
        return polished, "punct"
    a_flat, b_flat = a.replace(" ", ""), b.replace(" ", "")
    if len(b_flat) / len(a_flat) < POLISH_MIN_LENGTH:
        return None, "length"
    keys = sorted({MATCH_STRIP.sub("", term).lower() for term in terms if len(MATCH_STRIP.sub("", term)) >= 2}, key=len, reverse=True)
    # 原文裡已有的術語不能被改掉
    for key in keys:
        if key in a_flat and key not in b_flat:
            return None, f"term {key}"
    if digit_runs(cleaned, keys) != digit_runs(checked, keys):
        return None, "number"
    pattern = NEGATION[language]
    if len(pattern.findall(b)) < len(pattern.findall(a)):
        return None, "negation"
    previous_tokens = polish_tokens(previous, language)
    fillers = filler_tokens(language)
    # 相鄰的非 equal 區段合併成一段再判斷（重講加贅詞、聽錯加大小寫都會拆成好幾個 opcode）
    regions = []
    chained = False
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_tokens, b_tokens, autojunk=False).get_opcodes():
        if tag == "equal":
            chained = False
        elif chained:
            regions[-1][1], regions[-1][3] = i2, j2
        else:
            regions.append([i1, i2, j1, j2])
            chained = True
    for i1, i2, j1, j2 in regions:
        start, end = i1, i2
        stripping = True
        while stripping and start < end:
            stripping = False
            for filler in fillers:
                size = len(filler)
                if tuple(a_tokens[start:start + size]) == filler and start + size <= end:
                    start += size
                    stripping = True
                    break
                if tuple(a_tokens[end - size:end]) == filler and end - size >= start:
                    end -= size
                    stripping = True
                    break
        inserted = b_tokens[j1:j2]
        if inserted:
            reason = "insert" if start == end else replacement_reason(a_tokens[start:end], inserted, language, keys)
        else:
            reason = deletion_reason(a_tokens, start, end, i1, i2, previous_tokens, language)
        if reason:
            return None, f"{reason}: {sep.join(a_tokens[i1:i2])}→{sep.join(inserted)}"
    return polished, "ok"


class Polisher:
    """翻譯前潤稿：用通用 LLM（Qwen3）把定稿原文整理乾淨，輸出仍是原文的語言。
    system prompt（規則、示例、術語表）整段預先算好 KV cache，每句只需處理前文與本句。"""

    def __init__(self, path, glossary):
        self.model, self.tokenizer = load(path)
        self.glossary = glossary
        self.terms = glossary.terms()
        self.sampler = make_sampler(temp=0.0)
        self.systems = {}
        self.caches = {}
        for language, template in POLISH_SYSTEM_PROMPTS.items():
            self.systems[language] = template.format(terms=("、" if language == "ja" else ", ").join(self.terms) or ("なし" if language == "ja" else "none"))
            self.caches[language] = self._cache(self.systems[language])
        # 先潤一句暖機：第一次推論要編譯，不先做的話第一句會撞逾時
        try:
            self._generate("ja", f"{POLISH_LABELS['ja'][0]}{POLISH_LABELS['ja'][2]}\n{POLISH_LABELS['ja'][1]}えっと明日は十五時で行きましょう", 24, POLISH_TIMEOUT_MAX * 2)
        except Exception as exc:
            note(f"polisher warm-up failed: {exc}")

    def _prompt(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _cache(self, system):
        full = self._prompt(system, PREFIX_PROBE)
        prefix = self._encode(full[: full.rfind(PREFIX_PROBE)])
        if self._encode(full)[: len(prefix)] != prefix:
            return None, None
        cache = make_prompt_cache(self.model)
        with gpu_lock:
            for _ in stream_generate(self.model, self.tokenizer, prompt=prefix, max_tokens=1, prompt_cache=cache):
                pass
            trim_prompt_cache(cache, cache[0].offset - len(prefix))
        return prefix, cache

    # 回傳 (輸出, 結果, 秒數)；結果是 ok、timeout 或 truncated。逾時從取得 GPU 後才開始算
    def _generate(self, language, user, max_tokens, timeout):
        tokens = self._encode(self._prompt(self.systems[language], user))
        prefix, cache = self.caches[language]
        if prefix and tokens[: len(prefix)] == prefix:
            tokens = tokens[len(prefix):]
        else:
            prefix, cache = None, None
        pieces = []
        outcome = "ok"
        with gpu_lock:
            started = time.monotonic()
            generator = stream_generate(self.model, self.tokenizer, prompt=tokens, max_tokens=max_tokens, prompt_cache=cache, sampler=self.sampler)
            try:
                for response in generator:
                    pieces.append(response.text)
                    if response.finish_reason == "length":
                        outcome = "truncated"
                    if time.monotonic() - started > timeout:
                        outcome = "timeout"
                        break
                    # 第一行寫完（出現換行且前面已有內容）就收：模型愛在後面補說明，整句失敗太可惜，只取第一行
                    text = "".join(pieces)
                    if "think>" not in text and "\n" in text.lstrip():
                        break
            finally:
                generator.close()
                if cache is not None:
                    trim_prompt_cache(cache, cache[0].offset - len(prefix))
            elapsed = time.monotonic() - started
        return "".join(pieces).split("</think>")[-1].strip(), outcome, elapsed

    # 回傳 (採用的文字或 None, 原因, 秒數, 模型原始輸出)
    def polish(self, text, language, context):
        context_label, text_label, none = POLISH_LABELS[language]
        user = f"{context_label}{' '.join(context) or none}\n{text_label}{text}"
        limit = len(text.split()) * 2 + 16 if language == "en" else len(text) + 16
        timeout = min(POLISH_TIMEOUT_MAX, POLISH_TIMEOUT + len(text) / POLISH_TIMEOUT_PER_CHARS)
        output, outcome, elapsed = self._generate(language, user, min(MAX_POLISH_TOKENS, limit), timeout)
        if outcome != "ok":
            return None, outcome, elapsed, output
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return None, "empty", elapsed, output
        polished = re.sub(r"^(?:潤稿|润稿|整えた文|Output|Cleaned)\s*[:：]\s*", "", lines[0]).strip().strip('"“”「」『』')
        if language == "ja":
            polished = JA_ASCII_SPACE.sub("", CJK_SPACE.sub("", polished))
        accepted, reason = accept_polish(text, polished, language, self.terms, context[-1] if context else "")
        return accepted, reason, elapsed, polished


class Finalizer:
    """每句用 Qwen3-ASR 重解該句音訊（有模型且收到音訊時），再刪贅詞、套術語表，送出 final（先不帶譯文）；
    有潤稿模型時再潤稿，文字有變就再送一次 final（逐字稿不用等潤稿），最後把文字交給 translator 翻譯。
    沒有重解模型或沒收到音訊時直接用 SpeechAnalyzer 的文字。"""

    def __init__(self, refiner, glossary, translator=None, polisher=None):
        self.refiner = refiner
        self.glossary = glossary
        self.translator = translator
        self.polisher = polisher
        # 依 utterance 記錄：recent 是定稿（潤稿後）的文字，潤稿時當前文；raw 是重解後、刪贅詞前的文字，用來切掉重解混進的上一句尾巴。
        # 同一句重送時覆蓋；取前文只取編號比目前小的（跟 Translator._context 一樣）
        self.recent = {}
        self.raw = {}
        if translator is not None:
            translator.finalizer = self
        self.jobs = queue.Queue()
        self.pending = 0
        self.pending_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, utterance, text, language, audio):
        with self.pending_lock:
            self.pending += 1
        self.jobs.put((utterance, text, language, audio))

    # 佇列裡有工作或正在處理；Translator 靠這個讓定稿優先
    def busy(self):
        with self.pending_lock:
            return self.pending > 0

    def close(self):
        self.jobs.put(None)
        self.thread.join()

    @staticmethod
    def _before(history, utterance, count):
        return [text for number, text in sorted(history.items()) if number < utterance][-count:]

    @staticmethod
    def _store(history, utterance, text):
        history[utterance] = text
        for number in sorted(history)[:-HISTORY_LIMIT]:
            del history[number]

    def _refine(self, utterance, text, language, audio):
        samples = np.frombuffer(audio, dtype="<f4").copy()
        started = time.monotonic()
        try:
            refined = self.refiner.transcribe(samples, language)
        except Exception as exc:
            note(f"refine failed utterance={utterance}: {exc}")
            return text
        reason = refine_reject_reason(text, refined, language)
        previous = self._before(self.raw, utterance, 1)
        chosen = text if reason else strip_repeated_head(choose_final(text, refined, language), previous[-1] if previous else "")
        suffix = f" rejected: {reason}" if reason else ("" if chosen == refined else f" (chose {chosen!r})")
        note(f"refine utterance={utterance} {language} {time.monotonic() - started:.2f}s audio={len(samples) / SAMPLE_RATE:.2f}s {text!r} -> {refined!r}{suffix}")
        return chosen

    def _polish(self, utterance, text, language):
        if self.polisher is None or not is_polishable(text, language):
            return text
        context = self._before(self.recent, utterance, CONTEXT_UTTERANCES)
        try:
            accepted, reason, elapsed, output = self.polisher.polish(text, language, context)
        except Exception as exc:
            note(f"polish failed utterance={utterance}: {exc}")
            return text
        if accepted is None:
            note(f"polish utterance={utterance} {language} {elapsed:.2f}s {text!r} -> {output!r} rejected: {reason}")
            return text
        accepted = self.glossary.apply(accepted, language)
        note(f"polish utterance={utterance} {language} {elapsed:.2f}s {text!r} -> {accepted!r}")
        return accepted

    def _process(self, utterance, text, language, audio):
        if language not in REFINE_LANGUAGE:
            language = language_of(text)
        short = len(audio) < (MIN_REFINE_SECONDS + REFINE_PADDING_SECONDS) * BYTES_PER_SECOND
        if self.refiner is not None and audio and not (short and language == "ja"):
            text = self._refine(utterance, text, language, audio)
        self._store(self.raw, utterance, text)
        final = self.glossary.apply(remove_fillers(text, language), language)
        emit("final", utterance, final)
        polished = self._polish(utterance, final, language)
        if polished != final:
            final = polished
            emit("final", utterance, final)
        self._store(self.recent, utterance, final)
        if self.translator is not None:
            self.translator.submit(utterance, final, language)

    # 單句出錯不能讓執行緒死掉：警告後至少用原文送 final，翻譯照常
    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            utterance, text, language, _ = job
            try:
                self._process(*job)
            except Exception as exc:
                note(f"finalize failed utterance={utterance}: {exc}")
                emit("warning", utterance, message=f"定稿處理失敗，改用原文：{exc}", code="finalize_failed")
                try:
                    emit("final", utterance, text)
                    if self.translator is not None:
                        self.translator.submit(utterance, text, language)
                except Exception as inner:
                    note(f"finalize fallback failed utterance={utterance}: {inner}")
            finally:
                with self.pending_lock:
                    self.pending -= 1


def load_translator(path, mode, glossary):
    if not path or not os.path.isdir(path):
        emit("warning", message="找不到翻譯模型，只出逐字稿", code="translator_load")
        return None
    try:
        return Translator(path, mode, glossary)
    except Exception as exc:
        note(f"translator load failed: {exc}")
        emit("warning", message=f"翻譯模型載入失敗，只出逐字稿：{exc}", code="translator_load")
        return None


def load_polisher(path, glossary):
    if not os.path.isdir(path):
        emit("warning", message="找不到潤稿模型，改用未潤稿的原文", code="polish_load")
        return None
    try:
        return Polisher(path, glossary)
    except Exception as exc:
        note(f"polisher load failed: {exc}")
        emit("warning", message=f"潤稿模型載入失敗，改用未潤稿的原文：{exc}", code="polish_load")
        return None


def transcribe_only(models_dir, refiner_path, translation_path, mode, polish_path=None):
    emit("status", message="準備中")
    glossary = Glossary()
    path = glossary_path(models_dir)
    if path:
        try:
            glossary.load(path)
        except OSError as exc:
            note(f"glossary load failed: {exc}")
        note(f"glossary: {len(glossary.entries)} replacements, {len(glossary.terms())} terms from {path}")
    refiner = None
    if refiner_path and not os.path.isdir(refiner_path):
        emit("warning", message="找不到 Qwen3-ASR 模型，改用 Apple 辨識原文", code="refine_load")
    elif refiner_path:
        try:
            refiner = Refiner(refiner_path)
        except Exception as exc:
            note(f"refiner load failed: {exc}")
            emit("warning", message=f"Qwen 重解載入失敗，改用 Apple 辨識原文：{exc}", code="refine_load")
    polisher = load_polisher(polish_path, glossary) if polish_path else None
    translator = load_translator(translation_path, mode, glossary) if translation_path else None
    note(f"transcribe refiner={refiner_path if refiner else None} polisher={polish_path if polisher else None} translator={translation_path if translator else None} target={mode}")
    finalizer = Finalizer(refiner, glossary, translator, polisher)
    emit("ready", message="就緒")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except ValueError:
            continue
        text = str(job.get("text", "")).strip()
        if text:
            try:
                audio = base64.b64decode(job["audio"]) if job.get("audio") else b""
            except ValueError as exc:
                note(f"bad audio utterance={job.get('utterance')}: {exc}")
                audio = b""
            finalizer.submit(int(job.get("utterance", 0)), text, job.get("language") or language_of(text), audio)
    finalizer.close()
    if translator is not None:
        translator.close()
    return 0


USAGE = "用法：worker.py --transcribe <Models 目錄> [--refine <Qwen3-ASR 路徑>] [--polish <Qwen3 路徑>] [--translate <PLaMo 路徑> --target <zh-hant|ja-en>]"


def parse_arguments(argv):
    options = {"--transcribe": None, "--refine": None, "--polish": None, "--translate": None, "--target": "zh-hant"}
    index = 0
    while index < len(argv):
        if argv[index] not in options or index + 1 >= len(argv):
            return None
        options[argv[index]] = argv[index + 1]
        index += 2
    if options["--transcribe"] is None or options["--target"] not in TARGETS:
        return None
    return options


def main():
    options = parse_arguments(sys.argv[1:])
    if options is None:
        emit("error", message=USAGE)
        return 2
    return transcribe_only(options["--transcribe"], options["--refine"], options["--translate"], options["--target"], options["--polish"])


if __name__ == "__main__":
    raise SystemExit(main())
