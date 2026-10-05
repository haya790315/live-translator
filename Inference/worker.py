import difflib
import json
import os
import re
import sys
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import mlx.core as mx
try:
    import onnxruntime
except ImportError:
    onnxruntime = None
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel


SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 4
VOICE_THRESHOLD = 0.0015
VAD_FRAME_SAMPLES = 512
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 4
VAD_FRAME_SECONDS = VAD_FRAME_SAMPLES / SAMPLE_RATE
VAD_ON = 0.5
VAD_OFF = 0.35
MIN_CONFIDENCE = 0.8
PRE_ROLL_BYTES = BYTES_PER_SECOND // 2
TAIL_BYTES = int(BYTES_PER_SECOND * 0.35)
MIN_PARTIAL_BYTES = int(BYTES_PER_SECOND * 0.6)
MIN_VOICED_SECONDS = 0.2
MIN_NEW_BYTES = int(BYTES_PER_SECOND * 0.3)
MAX_UTTERANCE_BYTES = BYTES_PER_SECOND * 12
MAX_BUFFER_BYTES = BYTES_PER_SECOND * 16
SHORT_SILENCE_SECONDS = 0.5
SHORT_SILENCE_BYTES = int(BYTES_PER_SECOND * SHORT_SILENCE_SECONDS)
LONG_SILENCE_SECONDS = 1.6
LONG_SILENCE_BYTES = int(BYTES_PER_SECOND * LONG_SILENCE_SECONDS)
CONTINUATION_SILENCE_SECONDS = 3.0
CONTINUATION_SILENCE_BYTES = int(BYTES_PER_SECOND * CONTINUATION_SILENCE_SECONDS)
CONTEXT_UTTERANCES = 2
SETTLE_SECONDS = 0.8
CUT_MARGIN_SECONDS = 0.2
REFINE_MARGIN_SECONDS = 0.1
REFINE_MATCH = 0.5
MIN_FORCED_CUT_SECONDS = 4.0
PAUSE_HOLD_SECONDS = 0.35
PARTIAL_INTERVAL = 0.4
PARTIAL_INTERVAL_BUSY = 1.5
MAX_TRANSLATION_TOKENS = 160
MAX_REFINE_TOKENS = 200
REFINER_LANGUAGES = {"Japanese": "ja", "English": "en"}
REFINER_NAMES = {"ja": "Japanese", "en": "English"}
REFINER_AUTO_SECONDS = 2.0
REPETITION_PENALTY = 1.05
MEMORY_LIMIT = 300
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])|(?<=です|ます)(?![がのねよかしけでとらっ。！？!?、])|(?<=ません|でした|ました)(?![がのねよかしけでとらっ。！？!?、])")
SENTENCE_END = re.compile(r"(?:[。！？!?]|です|ます|ました|ません|でした|ください|でしょう|ましょう|ですね|ますね|ですよ|ますよ)$")
SENTENCE_END_EN = re.compile(r"[.?!]$")
CONTINUATION_END_EN = re.compile(r"(?:,|\b(?:and|but|so|because|or|to|of|the|a|an|with|for|in|on|at|that|which|if|when|then|as|by|from|into|about|than|while|although|though|since|until|unless|whether)|-)$", re.IGNORECASE)
LATIN = re.compile(r"[A-Za-z]")
JAPANESE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
KATAKANA = re.compile(r"[\u30a0-\u30ffー]")
LANGUAGE_MARGIN = 0.03
CHARS_PER_VOICED_SECOND = {"ja": 6.0, "en": 12.0}
SHORT_OUTPUT_RATIO = 0.45
CONTINUATION_END = re.compile(r"(?:[、，,・0-9０-９]|[はがをにでとのへもや]|から|まで|けど|ので|って|たり|し|て|そして|また|それから|および|あるいは|今後|次に|まず)$")
NO_SPLIT_BEFORE = "がのねよかしけでとらっ、。！？!?"
TOKEN_GAP_SECONDS = 0.35
JUDGE_COMPLETE = 0.8
JUDGE_INCOMPLETE = 0.2
judge = None
MAX_DUPLICATE_CHARS = 6
DEBUG = os.environ.get("LIVE_TRANSLATOR_DEBUG") == "1"
HALLUCINATIONS = {"ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録お願いします", "チャンネル登録をお願いします", "字幕視聴ありがとうございました"}
FILLERS = {"えっと", "えーと", "ええと", "えー", "え", "あー", "あ", "あの", "あのー", "うーん", "うん", "ん", "んー", "んん", "ピッ", "はあ", "ふー", "まあ", "um", "uh", "uhm", "umm", "hmm", "hm", "mm", "mhm", "er", "ah", "oh", "huh"}
FILLER_SPLIT = re.compile(r"[、。！？!?…ー\s.,]+")
NORMALIZE = re.compile(r"[\s。、！？!?…]")
MATCH_STRIP = re.compile(r"[\s。、，,.！？!?…「」『』\"'’]")
REFINE_SPLIT = {"ja": re.compile(r"(?<=[。！？!?])"), "en": re.compile(r"(?<=[.?!])\s+")}
CJK_SPACE = re.compile(r"(?<=[^\x00-\x7f])[ \u3000]+(?=[^\x00-\x7f])|^[ \u3000]+|[ \u3000]+$")

HUNYUAN_PREFIX = "<|startoftext|>把下面的文本翻译成简体中文，不要额外解释。\n\n"
HUNYUAN_SUFFIX = "<|extra_0|>"
HY_MT2_RULES = "保持口语、简洁、忠实，不要补充原文没有的内容，不要解释。"
PREFIX_PROBE = "テスト"
KANA = re.compile(r"[\u3040-\u30ff]")
CHAT_SYSTEM_PROMPT = """你是实时会议字幕的日译中译者。把用户输入的日文翻成自然、准确的简体中文。
规则：
- 只输出译文。不要解释、不要注释、不要重复原文、不要加引号。
- 数字、金额、日期、否定、人名、公司名、产品名、英文缩写都必须正确保留。
- 「円」译为「日元」。
- 人名后的「さん」「様」等敬称省略，只保留人名。
- 输出不可含有任何日文假名。
示例：
日文：来週の金曜日までに、新しいAPIの設計書を提出してください。
译文：请在下周五之前提交新 API 的设计书。
日文：田中さんはこの件を担当しません。
译文：田中不负责这件事。"""

# 潤稿：定稿後、翻譯前，用通用 LLM（預設 Qwen3-4B）把辨識出的原文整理乾淨，輸出仍是原文的語言
POLISH_CONTEXT = 3
POLISH_MIN_CHARS = 7
# 潤稿在翻譯之後、佇列空著時才跑，沒有人在等它，所以上限可以放寬；新句子一到就中止
POLISH_TIMEOUT = 6.0
MAX_POLISH_TOKENS = 120
POLISH_MIN_LENGTH = 0.6
POLISH_LINE = re.compile(r"润稿\s*[:：]\s*(.*)")
LATIN_WORD = re.compile(r"[A-Za-z]+")
PARTICLE_START = re.compile(r"^[とがをにはも](?=[^\x00-\x7f])")
POLISH_SYSTEM_PROMPT = """你是会议逐字稿的校对员。对【本句】做最小限度的修正，输出与输入同一种语言，绝不翻译。
只允许做这四件事：
1. 删掉填充词（えー、あの、えっと、um、uh）和说错后重说的重复片段
2. 【本句】开头若重复了【前文】最后一句的结尾，删掉重复的开头
3. 依【术语表】和前后文，把语音识别听错的专有名词改成正确写法
4. 补上标点
禁止：改写措辞、调换语序、删除任何名词或动词、添加原文没有的内容、把疑问句改成陈述句。
禁止改动语尾和助词：思ってます 不要改成 思っています，ので 不要改成 けど，口语保持口语。
拿不准就原样输出。只输出一行，格式：
润稿：<修正后的句子>

示例一
【前文】无
【本句】えっとあのサーバーの移行はあの来週の金曜日までに完了する予定です
润稿：サーバーの移行は来週の金曜日までに完了する予定です。

示例二
【术语表】Shopify（误听：ショッピファイ）
【前文】次はショッピファイの件ですね。
【本句】件ですねショピファイの決済まわりでちょっと確認したいことがあります
润稿：Shopifyの決済まわりでちょっと確認したいことがあります。

示例三
【前文】无
【本句】So, um, we we should probably sync with the the backend team first.
润稿：So, we should probably sync with the backend team first."""

# 會議記憶：翻譯佇列空著時，從最近幾句原始文字抽出主題與聽錯的專有名詞，供潤稿與翻譯參考
MEMORY_EVERY = 6
MEMORY_IDLE_SECONDS = 3.0
MEMORY_WINDOW = 8
MEMORY_TIMEOUT = 8.0
MAX_MEMORY_TOKENS = 160
GLOSSARY_LIMIT = 40
TOPIC_LIMIT = 60
MEMORY_TOPIC_LINE = re.compile(r"主题\s*[:：]\s*(.*)")
MEMORY_TERMS_LINE = re.compile(r"术语\s*[:：]\s*(.*)")
TERM_SPLIT = re.compile(r"\s*[、,，;；]\s*")
MEMORY_SYSTEM_PROMPT = """你是会议记录助手。根据【最近的逐字稿】（语音识别的原始输出，专有名词可能听错）更新会议信息。
输出固定两行：
主题：<一句话概括【最近的逐字稿】正在讨论什么，不超过 40 字；话题变了就换>
术语：<新发现的专有名词，格式 听错写法=正确写法，用、分隔；只列人名、公司名、产品名、服务名、英文缩写；只列【当前术语表】里没有的；没有写 无>
规则：
- 听错写法必须一字不差地出现在【最近的逐字稿】里，包括用片假名拼出来的外来名称（如 ショピファイト=Shopify）和拼错的英文（如 sopifi=Shopify）
- 正确写法用通行的正式写法（如 Shopify、LINE、GitHub）；拿不准的不要列
- 不要解释"""

emit_lock = threading.Lock()
gpu_lock = threading.Lock()


def emit(kind, utterance=0, japanese="", chinese="", message="", **extra):
    payload = {"kind": kind, "utterance": utterance, "japanese": japanese, "chinese": chinese, "message": message}
    payload.update(extra)
    with emit_lock:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def debug(message):
    if DEBUG:
        sys.stderr.write(f"[worker {time.monotonic():.2f}] {message}\n")
        sys.stderr.flush()


# 不受 DEBUG 控制，App 會把 stderr 接到 Logs/app.log；用來留下潤稿前後的對照
def note(message):
    sys.stderr.write(f"[worker] {message}\n")
    sys.stderr.flush()


class Judge:
    def __init__(self, path):
        spec = json.load(open(path))
        self.suffix = spec["suffix"]
        self.max_n = spec["max_n"]
        self.buckets = spec["buckets"]
        self.bias = spec["bias"]
        self.weights = {int(key): value for key, value in spec["weights"].items()}

    def _features(self, text):
        tail = text[-self.suffix:]
        keys = [f"len{min(len(text) // 4, 10)}"]
        if text and text[-1] in "。？！?!":
            keys.append("punct")
        for n in range(1, self.max_n + 1):
            for start in range(max(0, len(tail) - n - 3), len(tail) - n + 1):
                keys.append(f"{n}:{tail[start:start + n]}:{len(tail) - start - n}")
        return keys

    def _bucket(self, key):
        value = 2166136261
        for char in key.encode("utf-8"):
            value = ((value ^ char) * 16777619) & 0xFFFFFFFF
        return value % self.buckets

    def __call__(self, text):
        total = self.bias + sum(self.weights.get(self._bucket(key), 0.0) for key in self._features(text))
        return 1.0 / (1.0 + np.exp(-total))


def load_judge(path):
    if not os.path.isfile(path):
        debug("segment judge unavailable, using rules only")
        return None
    return Judge(path)


def is_complete(text, allow_short=False, language="ja"):
    text = text.strip().rstrip("」』")
    if not text:
        return False
    if language == "en":
        if CONTINUATION_END_EN.search(text):
            return False
        if SENTENCE_END_EN.search(text):
            return True
        return allow_short and len(text.split()) <= 3
    if judge is not None and len(text) >= 3:
        probability = judge(text)
        if probability >= JUDGE_COMPLETE:
            return True
        if probability <= JUDGE_INCOMPLETE:
            return False
    if CONTINUATION_END.search(text):
        return False
    if SENTENCE_END.search(text):
        return True
    return allow_short and len(text) <= 4


def is_continuation(text, language="ja"):
    text = text.strip().rstrip("」』")
    if language == "en":
        return bool(CONTINUATION_END_EN.search(text))
    if judge is not None and len(text) >= 3 and judge(text) <= JUDGE_INCOMPLETE:
        return True
    return bool(CONTINUATION_END.search(text))


def strip_duplicate_tail(segments, last_final):
    if not segments or not last_final:
        return segments
    head = NORMALIZE.sub("", segments[0][2])
    if 0 < len(head) <= MAX_DUPLICATE_CHARS and NORMALIZE.sub("", last_final).endswith(head):
        debug(f"drop duplicate tail {segments[0][2]!r}")
        return segments[1:]
    return segments


def common_prefix(a, b):
    size = min(len(a), len(b))
    index = 0
    while index < size and a[index] == b[index]:
        index += 1
    return a[:index]


class Vad:
    def __init__(self, path):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)
        self.rate = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, frame):
        window = np.concatenate([self.context, frame]).reshape(1, -1).astype(np.float32)
        output, self.state = self.session.run(None, {"input": window, "state": self.state, "sr": self.rate})
        self.context = frame[-64:]
        return float(output[0, 0])


def load_vad(path):
    if onnxruntime is None or not os.path.isfile(path):
        debug("silero vad unavailable, falling back to energy threshold")
        return None
    return Vad(path)


def is_filler(text):
    pieces = [piece.lower() for piece in FILLER_SPLIT.split(text) if piece]
    return bool(pieces) and all(piece in FILLERS for piece in pieces)


def language_of(text):
    return "en" if len(LATIN.findall(text)) > len(JAPANESE.findall(text)) else "ja"


def strip_terminal(text, language):
    return text if language == "en" else text.rstrip("。")


def purity(text, language):
    if language != "ja":
        return 1.0
    japanese = len(JAPANESE.findall(text))
    if japanese < 4:
        return 1.0
    ratio = len(KATAKANA.findall(text)) / japanese
    return 1.0 - 0.6 * max(0.0, ratio - 0.5) / 0.5


class Snapshot:
    def __init__(self, audio, start_byte, total_bytes, voice_end_byte, last_voice, speaking, closed, gap):
        self.audio = audio
        self.start_byte = start_byte
        self.total_bytes = total_bytes
        self.voice_end_byte = voice_end_byte
        self.last_voice = last_voice
        self.speaking = speaking
        self.closed = closed
        self.gap = gap


class AudioState:
    def __init__(self, vad=None):
        self.lock = threading.Lock()
        self.vad = vad
        self.pending = np.zeros(0, dtype=np.float32)
        self.frames = []
        self.frame_voiced = False
        self.buffer = bytearray()
        self.start_byte = 0
        self.total_bytes = 0
        self.voice_end_byte = 0
        self.last_voice = 0.0
        self.speaking = False
        self.closed = False
        self.gaps = []

    def _frame_voiced(self, frame):
        if self.vad is None:
            return float(np.sqrt(np.mean(frame * frame))) >= VOICE_THRESHOLD
        probability = self.vad(frame)
        self.frame_voiced = probability >= VAD_ON or (self.frame_voiced and probability >= VAD_OFF)
        return self.frame_voiced

    def append(self, chunk):
        samples = np.concatenate([self.pending, np.frombuffer(chunk, dtype="<f4")])
        usable = len(samples) - len(samples) % VAD_FRAME_SAMPLES
        frames = samples[:usable].reshape(-1, VAD_FRAME_SAMPLES) if usable else samples[:0].reshape(0, VAD_FRAME_SAMPLES)
        self.pending = samples[usable:]
        voiced = [self._frame_voiced(frame) for frame in frames]
        with self.lock:
            self.buffer.extend(chunk)
            for index, flag in enumerate(voiced):
                frame_start = self.total_bytes + index * VAD_FRAME_BYTES
                self.frames.append((frame_start, flag))
                if flag:
                    if self.speaking and frame_start - self.voice_end_byte >= SHORT_SILENCE_BYTES:
                        self.gaps.append((self.voice_end_byte, frame_start))
                    self.last_voice = time.monotonic()
                    self.voice_end_byte = frame_start + VAD_FRAME_BYTES
                    self.speaking = True
            self.total_bytes += len(chunk)
            limit = MAX_BUFFER_BYTES if self.speaking else PRE_ROLL_BYTES
            if len(self.buffer) > limit:
                self._drop(len(self.buffer) - limit)

    def _drop(self, count):
        del self.buffer[:count]
        self.start_byte += count
        while self.frames and self.frames[0][0] + VAD_FRAME_BYTES <= self.start_byte:
            self.frames.pop(0)

    def voiced_seconds(self, start_byte, end_byte):
        with self.lock:
            count = sum(1 for frame_start, flag in self.frames if flag and frame_start + VAD_FRAME_BYTES > start_byte and frame_start < end_byte)
        return count * VAD_FRAME_SECONDS

    def close(self):
        with self.lock:
            self.closed = True

    def snapshot(self):
        with self.lock:
            gap = self.gaps[0] if self.gaps else None
            return Snapshot(bytes(self.buffer), self.start_byte, self.total_bytes, self.voice_end_byte, self.last_voice, self.speaking, self.closed, gap)

    def drop_gap(self, gap):
        with self.lock:
            if gap in self.gaps:
                self.gaps.remove(gap)

    def finish(self, end_byte):
        with self.lock:
            self._drop(min(max(end_byte - self.start_byte, 0), len(self.buffer)))
            self.gaps = [gap for gap in self.gaps if gap[0] >= end_byte]
            if self.voice_end_byte <= end_byte:
                self.speaking = False


def read_audio(state):
    remainder = b""
    while True:
        data = os.read(sys.stdin.fileno(), 65536)
        if not data:
            state.close()
            return
        joined = remainder + data
        usable = len(joined) - len(joined) % 4
        if usable:
            state.append(joined[:usable])
        remainder = joined[usable:]


def build_segments(tokens, language):
    segments = []
    current = []
    for index, token in enumerate(tokens):
        current.append(token)
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        text = CJK_SPACE.sub("", "".join(item.text for item in current)).strip()
        boundary = following is None or following.start - token.end >= TOKEN_GAP_SECONDS
        if not boundary:
            if language == "en":
                boundary = bool(SENTENCE_END_EN.search(text))
            elif SENTENCE_END.search(text):
                boundary = not following.text.strip().startswith(tuple(NO_SPLIT_BEFORE))
        if boundary:
            if text:
                segments.append((float(current[0].start), float(token.end), text))
            current = []
    return segments


def join_segments(segments, language):
    return (" " if language == "en" else "").join(segment[2] for segment in segments)


def decode_candidate(model, samples, language, voiced):
    result = model.generate(get_logmel(samples, model.preprocessor_config))[0]
    tokens = [token for sentence in result.sentences for token in sentence.tokens]
    if not tokens:
        return {"language": language, "text": "", "segments": [], "score": 0.0, "confidence": 0.0}
    confidence = float(np.mean([float(token.confidence) for token in tokens]))
    segments = build_segments(tokens, language)
    text = join_segments(segments, language)
    expected = voiced * CHARS_PER_VOICED_SECOND[language]
    coverage = min(1.0, len(NORMALIZE.sub("", text)) / max(1.0, expected * SHORT_OUTPUT_RATIO))
    script = 1.0 if language_of(text) == language else 0.5
    return {"language": language, "text": text, "segments": segments, "confidence": confidence, "score": confidence * coverage * script * purity(text, language)}


def transcribe(audio, models, voiced, preferred=None):
    if voiced < MIN_VOICED_SECONDS:
        return "", [], preferred
    samples = mx.array(np.frombuffer(audio, dtype="<f4").copy())
    with gpu_lock:
        candidates = [decode_candidate(model, samples, language, voiced) for language, model in models.items() if model is not None]
    best = max(candidates, key=lambda candidate: candidate["score"])
    if preferred and preferred != best["language"]:
        current = next((candidate for candidate in candidates if candidate["language"] == preferred), None)
        if current is not None and best["score"] - current["score"] < LANGUAGE_MARGIN:
            best = current
    if len(candidates) > 1:
        debug("candidates " + " | ".join(f"{c['language']} score={c['score']:.2f} conf={c['confidence']:.2f} {c['text'][:40]!r}" for c in candidates) + f" -> {best['language']}")
    if best["confidence"] and best["confidence"] < MIN_CONFIDENCE:
        debug(f"drop low-confidence decode {best['confidence']:.2f} {best['text']!r}")
        return "", [], best["language"]
    text = best["text"]
    if NORMALIZE.sub("", text) in HALLUCINATIONS or is_filler(text):
        debug(f"drop filler or hallucination {text!r}")
        return "", [], best["language"]
    return text, best["segments"], best["language"]


class Refiner:
    def __init__(self, path):
        from mlx_audio.stt.utils import load as load_stt
        self.model = load_stt(path)
        self.model.generate(np.zeros(SAMPLE_RATE, dtype=np.float32), max_tokens=8)

    def transcribe(self, audio, language):
        samples = np.frombuffer(audio, dtype="<f4").copy()
        hint = REFINER_NAMES.get(language) if len(samples) < SAMPLE_RATE * REFINER_AUTO_SECONDS else None
        with gpu_lock:
            output = self.model.generate(samples, max_tokens=MAX_REFINE_TOKENS, language=hint)
        names = output.language if isinstance(output.language, list) else [output.language]
        language = REFINER_LANGUAGES.get(names[0] if names else "")
        text = output.text.strip()
        if language == "ja":
            text = CJK_SPACE.sub("", text)
        return text, language


def overlap(sentence, original):
    a = MATCH_STRIP.sub("", sentence).lower()
    b = MATCH_STRIP.sub("", original).lower()
    if not a or not b:
        return 0.0
    matched = sum(block.size for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if block.size >= 2)
    return matched / len(a)


def choose_final(original, refined, language):
    if not refined or language is None:
        return original
    if NORMALIZE.sub("", refined) in HALLUCINATIONS or is_filler(refined):
        return original
    sentences = [piece.strip() for piece in REFINE_SPLIT[language].split(refined) if piece.strip()]
    kept = [index for index, sentence in enumerate(sentences) if overlap(sentence, original) >= REFINE_MATCH]
    if not kept:
        return refined
    return (" " if language == "en" else "").join(sentences[kept[0]: kept[-1] + 1])


# 開頭子音的類別：聽錯的字和正確的字發音相近，開頭子音幾乎都保留；模型把對照配錯（ショピファイト=LINE）時靠這個擋
KATAKANA_ONSET = {
    "v": "アイウエオァィゥェォ", "k": "カキクケコガギグゲゴ", "s": "サシスセソザジズゼゾ", "t": "タチツテトダヂヅデド",
    "n": "ナニヌネノ", "h": "ハヒフヘホパピプペポ", "b": "バビブベボヴ", "m": "マミムメモ", "y": "ヤユヨ", "r": "ラリルレロ", "w": "ワヲ",
}
LATIN_ONSET = {
    "v": "aeiou", "k": "ckqg", "s": "szxj", "t": "td", "n": "n", "h": "hfp", "b": "bv", "m": "m", "y": "y", "r": "lr", "w": "w",
}


def onset_class(text):
    head = text[:1].lower()
    for table in (KATAKANA_ONSET, LATIN_ONSET):
        for name, members in table.items():
            if head in members:
                return name
    return None


def plausible_term(wrong, right):
    if not LATIN.match(right):
        return True
    if not (LATIN.match(wrong) or KATAKANA.match(wrong)):
        return True
    a, b = onset_class(wrong), onset_class(right)
    return a is None or b is None or a == b


class Glossary:
    """聽錯寫法 → 正確寫法。只收錄能在原始文字裡找到、開頭發音同類的對照，套用時不經過模型。"""

    def __init__(self):
        self.entries = {}

    def add(self, wrong, right, evidence):
        wrong, right = wrong.strip().strip("「」『』\"'"), right.strip().strip("「」『』\"'")
        # 正確寫法至少兩個字：模型拿不準時會亂猜一個字母（ヒグマ=H）；也擋掉「あと=あと（无）」這種把「无」填進欄位的輸出
        if len(wrong) < 2 or len(right) < 2 or len(right) > 30 or "=" in right or wrong == right:
            return False
        if re.search(r"[（()）]|无|無", right) or right.startswith(wrong) or wrong.startswith(right):
            return False
        if wrong.lower() not in evidence.lower() or wrong in self.entries or not plausible_term(wrong, right):
            return False
        if len(self.entries) >= GLOSSARY_LIMIT:
            del self.entries[next(iter(self.entries))]
        self.entries[wrong] = right
        return True

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
                text = text.replace(wrong, right)
        return text

    def terms(self):
        return list(dict.fromkeys(self.entries.values()))

    def describe(self):
        grouped = {}
        for wrong, right in self.entries.items():
            grouped.setdefault(right, []).append(wrong)
        return "、".join(f"{right}（误听：{'、'.join(wrongs)}）" for right, wrongs in grouped.items())


FILLERS_BY_LENGTH = sorted(FILLERS, key=len, reverse=True)


def is_filler_run(segment):
    """整段都由語氣詞拼成（えーとあの、umuh）。"""
    segment = segment.lower()
    while segment:
        for filler in FILLERS_BY_LENGTH:
            if segment.startswith(filler):
                segment = segment[len(filler):]
                break
        else:
            return False
    return True


# 准刪的只有三種：語氣詞、緊鄰的重講（we can, we can）、重複了上一句結尾的開頭
def allowed_deletion(a, i1, i2, previous):
    segment = a[i1:i2]
    if is_filler_run(segment):
        return True
    if a[i2:].startswith(segment) or a[:i1].endswith(segment):
        return True
    return i1 == 0 and bool(previous) and previous.endswith(segment)


def accept_polish(original, polished, language, terms=(), previous=""):
    """潤稿只准刪贅詞、刪重講、加標點，不准改字、不准加字；判定不過就沿用原文。
    專有名詞的修正在這之前已由術語表做掉，所以不需要放行替換。回傳 (採用的文字或 None, 原因)。"""
    if not polished or polished == original:
        return None, "same"
    if language == "ja" and not KANA.search(polished):
        return None, "language"
    if language == "en" and JAPANESE.search(polished):
        return None, "language"
    # 刪贅詞後殘留的助詞（えーとあの僕 → と僕）；原文本來就這個字開頭（はい）不算
    if language == "ja" and PARTICLE_START.match(polished) and not original.startswith(polished[0]):
        return None, "particle"
    # 英文字只有術語表裡的才准改大小寫（sopifi → SOPIFI 會被翻譯當成縮寫照抄）；其他改回原樣
    original_words = {word.lower(): word for word in LATIN_WORD.findall(original)}
    for word in set(LATIN_WORD.findall(polished)):
        if word.lower() in original_words and word != original_words[word.lower()] and word not in terms:
            polished = re.sub(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", original_words[word.lower()], polished)
    a = MATCH_STRIP.sub("", original)
    b = MATCH_STRIP.sub("", polished)
    if not a or not b:
        return None, "empty"
    if a == b:
        return polished, "punct"
    if len(b) / len(a) < POLISH_MIN_LENGTH:
        return None, "length"
    lowered = a.lower()
    previous = MATCH_STRIP.sub("", previous).lower()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, lowered, b.lower(), autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "delete" or not allowed_deletion(lowered, i1, i2, previous):
            return None, f"{tag} {a[i1:i2]}→{b[j1:j2]}"
    return polished, "ok"


class Polisher:
    """用通用 LLM 潤稿與抽會議記憶。兩段 system prompt 各自預先算好 KV cache。"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = make_sampler(temp=0.0)
        self.polish_prefix, self.polish_cache = self._cache(POLISH_SYSTEM_PROMPT)
        self.memory_prefix, self.memory_cache = self._cache(MEMORY_SYSTEM_PROMPT)

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

    # 回傳 (輸出, 結果, 秒數)；結果是 ok、timeout 或 preempted（should_stop 回傳 True，例如有新句子要翻）
    def _generate(self, system, prefix, cache, user, max_tokens, timeout, should_stop=None):
        tokens = self._encode(self._prompt(system, user))
        if prefix and tokens[: len(prefix)] == prefix:
            tokens = tokens[len(prefix):]
        else:
            cache = None
        started = time.monotonic()
        pieces = []
        outcome = "ok"
        generator = stream_generate(self.model, self.tokenizer, prompt=tokens, max_tokens=max_tokens, prompt_cache=cache, sampler=self.sampler)
        try:
            while True:
                # 每個 token 之間放開 GPU，parakeet 的即時解碼與翻譯才插得進來
                with gpu_lock:
                    try:
                        response = next(generator)
                    except StopIteration:
                        break
                pieces.append(response.text)
                if should_stop is not None and should_stop():
                    outcome = "preempted"
                    break
                if time.monotonic() - started > timeout:
                    outcome = "timeout"
                    break
        finally:
            generator.close()
            if cache is not None:
                with gpu_lock:
                    trim_prompt_cache(cache, cache[0].offset - len(prefix))
        return "".join(pieces).split("</think>")[-1].strip(), outcome, time.monotonic() - started

    def polish(self, text, language, context, topic, glossary, should_stop=None):
        user = f"【会议主题】{topic or '无'}\n【术语表】{glossary.describe() or '无'}\n【前文】{' / '.join(context) or '无'}\n【本句】{text}"
        output, outcome, elapsed = self._generate(POLISH_SYSTEM_PROMPT, self.polish_prefix, self.polish_cache, user, MAX_POLISH_TOKENS, POLISH_TIMEOUT, should_stop)
        if outcome != "ok":
            return None, outcome, elapsed
        match = next((POLISH_LINE.match(line) for line in output.splitlines() if POLISH_LINE.match(line)), None)
        if match is None:
            return None, "format", elapsed
        polished = match.group(1).strip().strip('"“”「」『』')
        if language == "ja":
            polished = CJK_SPACE.sub("", polished)
        accepted, reason = accept_polish(text, polished, language, glossary.terms(), context[-1] if context else "")
        return accepted, reason, elapsed

    def update_memory(self, recent, topic, glossary, should_stop=None):
        user = f"【当前主题】{topic or '无'}\n【当前术语表】{glossary.describe() or '无'}\n【最近的逐字稿】\n" + "\n".join(recent)
        output, outcome, elapsed = self._generate(MEMORY_SYSTEM_PROMPT, self.memory_prefix, self.memory_cache, user, MAX_MEMORY_TOKENS, MEMORY_TIMEOUT, should_stop)
        if outcome != "ok":
            # 被中途截斷的最後一行可能是寫到一半的術語，丟掉
            output = output.rsplit("\n", 1)[0] if "\n" in output else ""
        added = []
        evidence = "\n".join(recent)
        for line in output.splitlines():
            if match := MEMORY_TOPIC_LINE.match(line):
                candidate = match.group(1).strip()
                if candidate and candidate not in ("无", "無"):
                    topic = candidate[:TOPIC_LIMIT]
            elif match := MEMORY_TERMS_LINE.match(line):
                for pair in TERM_SPLIT.split(match.group(1)):
                    if "=" in pair:
                        wrong, right = pair.split("=", 1)
                        if glossary.add(wrong, right, evidence):
                            added.append(f"{wrong.strip()}={right.strip()}")
        return topic, added, outcome, elapsed


class Translator:
    def __init__(self, model, tokenizer, model_type="", refiner=None, name="", polisher=None):
        self.refiner = refiner
        self.polisher = polisher
        self.glossary = Glossary()
        self.topic = ""
        self.recent = []
        self.since_memory = 0
        self.last_final_time = 0.0
        self.model = model
        self.tokenizer = tokenizer
        template = getattr(tokenizer, "chat_template", None) or ""
        if "hy_User" in template or "hy-mt2" in name.lower():
            self.style = "hy_mt2"
        elif model_type.startswith("hunyuan"):
            self.style = "hunyuan_mt"
        else:
            self.style = "chat"
        self.hunyuan = self.style != "chat"
        self.context_aware = self.style in ("hy_mt2", "chat")
        self.history = []
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.jobs = []
        self.closed = False
        self.active = False
        self.memory = {}
        self.sampler = make_sampler(temp=0.0)
        self.logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY) if self.hunyuan else None
        self.prefix_tokens = None
        self.cache = None
        self._prepare_cache()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _prompt_text(self, japanese, context=(), topic="", terms=()):
        if self.style == "hunyuan_mt":
            return HUNYUAN_PREFIX + japanese + HUNYUAN_SUFFIX
        context_text = "".join(context)
        if self.style == "hy_mt2":
            tasks = []
            if context_text:
                tasks.append(f"这是会议的实时字幕，前文是：「{context_text}」，请结合前文理解语境，但不要翻译前文。")
            if topic:
                tasks.append(f"会议主题：{topic}。")
            if terms:
                tasks.append(f"术语表：{'、'.join(terms)}，译文中原样保留这些名词。")
            tasks.append(HY_MT2_RULES)
            tasks.append("将【待翻译文本】翻译为简体中文，只输出译文。")
            numbered = "\n".join(f"{index}、{task}" for index, task in enumerate(tasks, 1))
            messages = [{"role": "user", "content": f"【待翻译文本】\n{japanese}\n\n【翻译任务】\n{numbered}"}]
        else:
            system = CHAT_SYSTEM_PROMPT
            if context_text:
                system += f"\n前文（仅供理解语境，不要翻译）：{context_text}"
            if topic:
                system += f"\n会议主题：{topic}"
            if terms:
                system += f"\n术语表：{'、'.join(terms)}（译文中原样保留）"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": japanese},
            ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _prefix_text(self):
        text = self._prompt_text(PREFIX_PROBE)
        index = text.rfind(PREFIX_PROBE)
        return text[:index] if index > 0 else None

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prepare_cache(self):
        prefix_text = self._prefix_text()
        if not prefix_text:
            return
        prefix = self._encode(prefix_text)
        if self._encode(self._prompt_text(PREFIX_PROBE))[: len(prefix)] != prefix:
            return
        cache = make_prompt_cache(self.model)
        for _ in stream_generate(self.model, self.tokenizer, prompt=prefix, max_tokens=1, prompt_cache=cache):
            pass
        trim_prompt_cache(cache, cache[0].offset - len(prefix))
        self.prefix_tokens = prefix
        self.cache = cache

    def _pieces(self, japanese):
        return [piece.strip() for piece in SENTENCE_SPLIT.split(japanese) if piece.strip()]

    def submit(self, utterance, japanese, audio=b"", language="ja"):
        with self.lock:
            self.jobs.append((utterance, japanese, audio, language))
            self.wake.notify()

    def pending(self):
        with self.lock:
            return self.active or bool(self.jobs)

    def close(self):
        with self.lock:
            self.closed = True
            self.wake.notify()
        self.thread.join()

    def _remember(self, japanese, chinese):
        if len(self.memory) >= MEMORY_LIMIT:
            del self.memory[next(iter(self.memory))]
        self.memory[japanese] = chinese

    def _clean(self, text):
        text = text.split("</think>")[-1].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = lines[0] if lines else ""
        for prefix in ("译文：", "译文:", "翻译：", "翻译:", "譯文：", "譯文:", "翻譯：", "翻譯:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text.strip().strip('"“”「」『』')

    def _generate(self, japanese, context=()):
        chinese = self._generate_once(japanese, context)
        if not self.hunyuan and KANA.search(chinese):
            debug(f"kana in translation {chinese!r}, retrying")
            chinese = self._generate_once(japanese + "\n（请把上面的日文翻成简体中文，只输出中文译文）", context)
        return chinese

    def _generate_once(self, japanese, context=()):
        tokens = self._encode(self._prompt_text(japanese, context, self.topic, self.glossary.terms()))
        cache = None
        if self.prefix_tokens and tokens[: len(self.prefix_tokens)] == self.prefix_tokens:
            tokens = tokens[len(self.prefix_tokens):]
            cache = self.cache
        text = []
        with gpu_lock:
            try:
                for response in stream_generate(self.model, self.tokenizer, prompt=tokens, max_tokens=MAX_TRANSLATION_TOKENS, prompt_cache=cache, sampler=self.sampler, logits_processors=self.logits_processors):
                    text.append(response.text)
            finally:
                if cache is not None:
                    trim_prompt_cache(cache, cache[0].offset - len(self.prefix_tokens))
        return self._clean("".join(text))

    # 翻譯佇列空著、而且累積了 MEMORY_EVERY 句或閒置超過 MEMORY_IDLE_SECONDS 時更新會議記憶
    def _memory_due(self):
        if self.polisher is None or self.since_memory == 0:
            return False
        return self.since_memory >= MEMORY_EVERY or time.monotonic() - self.last_final_time >= MEMORY_IDLE_SECONDS

    # 有新句子要翻時中止背景工作（潤稿、會議記憶）讓路
    def _has_jobs(self):
        return bool(self.jobs)

    def _update_memory(self):
        recent = list(self.recent)
        self.since_memory = 0
        try:
            self.topic, added, outcome, elapsed = self.polisher.update_memory(recent, self.topic, self.glossary, self._has_jobs)
        except Exception as exc:
            note(f"memory failed: {exc}")
            return
        note(f"memory {elapsed:.2f}s {outcome} topic={self.topic!r} added={added}")

    # 翻譯前：套術語表（純字串替換，不經過模型）
    def _apply_glossary(self, utterance, text, language):
        self.recent = (self.recent + [text])[-MEMORY_WINDOW:]
        self.since_memory += 1
        applied = self.glossary.apply(text, language)
        if applied != text:
            emit("revised_japanese", utterance, applied)
        return applied

    # 翻譯後：LLM 潤稿。只在佇列空著時跑，新句子一到就中止，所以中文不會因為它變慢
    def _polish(self, utterance, text, language):
        if self.polisher is None or self._has_jobs():
            return
        if len(MATCH_STRIP.sub("", text)) < POLISH_MIN_CHARS:
            note(f"polish skip utterance={utterance} short {text!r}")
            return
        try:
            polished, reason, elapsed = self.polisher.polish(text, language, tuple(self.history[-POLISH_CONTEXT - 1:-1]), self.topic, self.glossary, self._has_jobs)
        except Exception as exc:
            polished, reason, elapsed = None, f"error {exc}", 0.0
        note(f"polish utterance={utterance} {elapsed:.2f}s {reason} {text!r} -> {polished!r}")
        if polished:
            emit("revised_japanese", utterance, polished)
            if self.history and self.history[-1] == text:
                self.history[-1] = polished

    def _run(self):
        while True:
            with self.lock:
                while not self.jobs and not self.closed and not self._memory_due():
                    self.wake.wait(timeout=0.5)
                if not self.jobs and self.closed:
                    return
                job = self.jobs.pop(0) if self.jobs else None
                # 背景工作不算 active，parakeet 的即時解碼才不會降速
                self.active = job is not None
            if job is None:
                self._update_memory()
                continue
            utterance, japanese, audio, language = job
            if self.refiner is not None and audio:
                started = time.monotonic()
                try:
                    refined, detected = self.refiner.transcribe(audio, language)
                except Exception as exc:
                    debug(f"refine failed: {exc}")
                    refined, detected = "", None
                chosen = choose_final(japanese, refined, detected)
                debug(f"refine utterance={utterance} {time.monotonic() - started:.2f}s audio={len(audio) / BYTES_PER_SECOND:.2f}s lang={language}->{detected} {japanese!r} -> {refined!r}{'' if chosen == refined else ' (kept original)'}")
                if chosen != japanese:
                    emit("revised_japanese", utterance, chosen)
                    japanese = chosen
            japanese = self._apply_glossary(utterance, japanese, language)
            started = time.monotonic()
            chinese = []
            if self.context_aware:
                try:
                    chinese.append(self._generate(japanese, tuple(self.history[-CONTEXT_UTTERANCES:])))
                except Exception as exc:
                    emit("error", message=str(exc))
            else:
                for piece in self._pieces(japanese):
                    translated = self.memory.get(piece)
                    if translated is None:
                        try:
                            translated = self._generate(piece)
                        except Exception as exc:
                            emit("error", message=str(exc))
                            translated = ""
                        else:
                            self._remember(piece, translated)
                    chinese.append(translated)
            self.history = (self.history + [japanese])[-max(CONTEXT_UTTERANCES, POLISH_CONTEXT) - 1:]
            emit("final", utterance, japanese, "".join(chinese))
            if self.polisher is not None:
                note(f"translate utterance={utterance} {time.monotonic() - started:.2f}s")
            # 閒置從翻完這句起算，背景更新才不會卡到緊接著的下一句
            self.last_final_time = time.monotonic()
            with self.lock:
                self.active = False
            self._polish(utterance, japanese, language)


# 選用元件的模型由 Models/<name>.txt 指定（環境變數優先）；內容為空或 none 視為關閉
def optional_model_path(models_dir, env_name, file_name):
    name = os.environ.get(env_name)
    if name is None:
        try:
            with open(os.path.join(models_dir, file_name)) as handle:
                name = handle.read().strip()
        except OSError:
            name = ""
    if not name or name == "none":
        return None
    path = name if os.path.isabs(name) else os.path.join(models_dir, name)
    return path if os.path.isdir(path) else None


# 潤稿模型和翻譯模型相同時共用一份權重
def load_polisher(path, translation_path, translation_model):
    if path is None:
        return None
    emit("status", message="準備中")
    if os.path.normpath(path) == os.path.normpath(translation_path):
        model, tokenizer = translation_model
    else:
        model, tokenizer = load(path)
    polisher = Polisher(model, tokenizer)
    debug(f"polisher: {path} cached={polisher.polish_prefix is not None}")
    return polisher


def translate_only(translation_path):
    if not os.path.isdir(translation_path):
        emit("error", message="尚未完成安裝")
        return 2
    emit("status", message="準備中")
    models_dir = os.path.dirname(os.path.normpath(translation_path))
    with open(os.path.join(translation_path, "config.json")) as handle:
        model_type = json.load(handle).get("model_type", "")
    model, tokenizer = load(translation_path)
    polisher = load_polisher(optional_model_path(models_dir, "LIVE_TRANSLATOR_POLISH_MODEL", "polish-model.txt"), translation_path, (model, tokenizer))
    translator = Translator(model, tokenizer, model_type=model_type, name=os.path.basename(os.path.normpath(translation_path)), polisher=polisher)
    debug(f"translate-only style={translator.style}")
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
            translator.submit(int(job.get("utterance", 0)), text, language=job.get("language") or language_of(text))
    translator.close()
    return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--translate":
        return translate_only(sys.argv[2])
    if len(sys.argv) not in (3, 4):
        emit("error", message="缺少本機模型路徑")
        return 2
    speech_path, translation_path = sys.argv[1:3]
    english_path = sys.argv[3] if len(sys.argv) == 4 else None
    if not os.path.isdir(speech_path) or not os.path.isdir(translation_path):
        emit("error", message="尚未完成安裝")
        return 2

    emit("status", message="準備中")
    models_dir = os.path.dirname(os.path.normpath(speech_path))
    vad = load_vad(os.path.join(models_dir, "silero-vad", "model.onnx"))
    global judge
    judge = load_judge(os.path.join(models_dir, "segmenter", "weights.json"))
    speech_models = {"ja": from_pretrained(speech_path)}
    if english_path and os.path.isdir(english_path):
        speech_models["en"] = from_pretrained(english_path)
    for model in speech_models.values():
        model.generate(get_logmel(mx.zeros((SAMPLE_RATE,)), model.preprocessor_config))
    debug(f"speech models: {list(speech_models)}")
    emit("status", message="準備中")
    with open(os.path.join(translation_path, "config.json")) as handle:
        model_type = json.load(handle).get("model_type", "")
    polish_path = optional_model_path(models_dir, "LIVE_TRANSLATOR_POLISH_MODEL", "polish-model.txt")
    refiner_path = optional_model_path(models_dir, "LIVE_TRANSLATOR_FINAL_MODEL", "final-model.txt")
    refiner = None
    if refiner_path and polish_path:
        # 兩個都開記憶體會超過 16 GB 機器能負荷的量；潤稿優先
        debug(f"refiner disabled because polisher is on")
    elif refiner_path:
        emit("status", message="準備中")
        refiner = Refiner(refiner_path)
    debug(f"refiner: {refiner_path if refiner else None}")
    model, tokenizer = load(translation_path)
    polisher = load_polisher(polish_path, translation_path, (model, tokenizer))
    translator = Translator(model, tokenizer, model_type=model_type, refiner=refiner, name=os.path.basename(os.path.normpath(translation_path)), polisher=polisher)
    debug(f"translator style={translator.style} prefix_cached={translator.prefix_tokens is not None}")
    emit("ready", message="就緒")

    state = AudioState(vad)
    threading.Thread(target=read_audio, args=(state,), daemon=True).start()
    utterance = 1
    last_partial_time = 0.0
    decoded_until = 0
    previous_partial = ""
    last_final = ""
    pause_checked = None
    pause_result = None
    language = "ja"

    while True:
        time.sleep(0.05)
        snap = state.snapshot()
        if snap.closed and (not snap.speaking or not snap.audio):
            translator.close()
            return 0
        if not snap.speaking:
            continue

        now = time.monotonic()
        pause = now - snap.last_voice
        if snap.closed:
            reason, upto = "close", snap.total_bytes
        elif snap.gap is not None:
            upto = min(snap.gap[0] + TAIL_BYTES, snap.gap[1])
            if snap.gap[1] - snap.gap[0] < CONTINUATION_SILENCE_BYTES and pause_checked == upto:
                state.drop_gap(snap.gap)
                continue
            reason = "gap"
        elif pause >= SHORT_SILENCE_SECONDS:
            upto = min(snap.voice_end_byte + TAIL_BYTES, snap.total_bytes)
            if pause_checked == upto and pause < CONTINUATION_SILENCE_SECONDS and (pause < LONG_SILENCE_SECONDS or is_continuation(previous_partial, language)):
                continue
            reason = "pause"
        elif len(snap.audio) >= MAX_UTTERANCE_BYTES:
            reason, upto = "max", snap.total_bytes
        else:
            interval = PARTIAL_INTERVAL_BUSY if translator.pending() else PARTIAL_INTERVAL
            if pause >= PAUSE_HOLD_SECONDS or snap.total_bytes - decoded_until < MIN_NEW_BYTES or now - last_partial_time < interval or len(snap.audio) < MIN_PARTIAL_BYTES:
                continue
            reason, upto = "partial", snap.total_bytes

        upto = max(upto, snap.start_byte)
        if pause_checked == upto and pause_result is not None:
            text, segments = pause_result
        else:
            try:
                text, segments, language = transcribe(snap.audio[: upto - snap.start_byte], speech_models, state.voiced_seconds(snap.start_byte, upto), language)
            except Exception as exc:
                emit("error", message=str(exc))
                text, segments = "", []
            segments = strip_duplicate_tail(segments, last_final)
            text = join_segments(segments, language)
        bare = strip_terminal(text, language)
        window = (upto - snap.start_byte) / BYTES_PER_SECOND
        debug(f"{reason} lang={language} window={window:.2f}s pause={pause:.2f}s judge={judge(bare) if judge and bare and language == 'ja' else None} segments={[(round(a, 2), round(b, 2), t) for a, b, t in segments]}")

        if reason == "close":
            finalize_all = True
        elif reason == "gap":
            limit = CONTINUATION_SILENCE_BYTES if is_continuation(bare, language) else LONG_SILENCE_BYTES
            finalize_all = snap.gap[1] - snap.gap[0] >= limit or is_complete(bare, allow_short=True, language=language)
        elif reason == "pause":
            later = state.snapshot()
            resumed = later.voice_end_byte > snap.voice_end_byte
            limit = CONTINUATION_SILENCE_SECONDS if is_continuation(bare, language) else LONG_SILENCE_SECONDS
            finalize_all = is_complete(bare, allow_short=True, language=language) or later.closed or (not resumed and time.monotonic() - later.last_voice >= limit)
        else:
            finalize_all = False

        cut_count = len(segments) if finalize_all else 0
        if not finalize_all:
            for index in range(len(segments) - 1, -1, -1):
                candidate = strip_terminal(segments[index][2], language) if index == len(segments) - 1 else segments[index][2]
                if is_complete(candidate, language=language) and segments[index][1] <= window - SETTLE_SECONDS:
                    cut_count = index + 1
                    break
            if cut_count == 0 and reason == "max":
                for index in range(len(segments) - 2, -1, -1):
                    if segments[index][1] >= MIN_FORCED_CUT_SECONDS:
                        cut_count = index + 1
                        break
                if cut_count == 0:
                    finalize_all = True
                    cut_count = len(segments)

        if finalize_all or cut_count:
            if finalize_all:
                final_text = text or previous_partial
                cut_byte = upto
                refine_byte = upto
                remainder = []
            else:
                final_text = join_segments(segments[:cut_count], language)
                cut_time = segments[cut_count - 1][1] + CUT_MARGIN_SECONDS
                refine_time = segments[cut_count - 1][1] + REFINE_MARGIN_SECONDS
                if cut_count < len(segments):
                    cut_time = min(cut_time, segments[cut_count][0])
                    refine_time = min(refine_time, segments[cut_count][0])
                cut_byte = snap.start_byte + int(round(cut_time * SAMPLE_RATE)) * 4
                refine_byte = snap.start_byte + int(round(refine_time * SAMPLE_RATE)) * 4
                remainder = segments[cut_count:]
            debug(f"finalize utterance={utterance} all={finalize_all} cut={(cut_byte - snap.start_byte) / BYTES_PER_SECOND:.2f}s text={final_text!r}")
            if final_text:
                emit("final_japanese", utterance, final_text)
                translator.submit(utterance, final_text, snap.audio[: refine_byte - snap.start_byte], language)
                last_final = final_text
            utterance += 1
            state.finish(cut_byte)
            pause_checked = None
            pause_result = None
            previous_partial = strip_terminal(join_segments(remainder, language), language)
            if previous_partial:
                emit("partial_japanese", utterance, previous_partial, stable="")
            decoded_until = upto
            last_partial_time = time.monotonic() if remainder else 0.0
        else:
            if reason == "gap":
                state.drop_gap(snap.gap)
            pause_checked = upto if reason in ("gap", "pause") else None
            pause_result = (text, segments) if reason == "pause" else None
            decoded_until = upto
            last_partial_time = time.monotonic()
            if bare and bare != previous_partial:
                stable = common_prefix(previous_partial, bare)
                previous_partial = bare
                emit("partial_japanese", utterance, bare, stable=stable)


if __name__ == "__main__":
    raise SystemExit(main())
