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


class Translator:
    def __init__(self, model, tokenizer, model_type="", refiner=None, name=""):
        self.refiner = refiner
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

    def _prompt_text(self, japanese, context=()):
        if self.style == "hunyuan_mt":
            return HUNYUAN_PREFIX + japanese + HUNYUAN_SUFFIX
        context_text = "".join(context)
        if self.style == "hy_mt2":
            tasks = []
            if context_text:
                tasks.append(f"这是会议的实时字幕，前文是：「{context_text}」，请结合前文理解语境，但不要翻译前文。")
            tasks.append(HY_MT2_RULES)
            tasks.append("将【待翻译文本】翻译为简体中文，只输出译文。")
            numbered = "\n".join(f"{index}、{task}" for index, task in enumerate(tasks, 1))
            messages = [{"role": "user", "content": f"【待翻译文本】\n{japanese}\n\n【翻译任务】\n{numbered}"}]
        else:
            system = CHAT_SYSTEM_PROMPT
            if context_text:
                system += f"\n前文（仅供理解语境，不要翻译）：{context_text}"
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
        tokens = self._encode(self._prompt_text(japanese, context))
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

    def _run(self):
        while True:
            with self.lock:
                while not self.jobs and not self.closed:
                    self.wake.wait()
                if not self.jobs:
                    return
                utterance, japanese, audio, language = self.jobs.pop(0)
                self.active = True
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
            chinese = []
            if self.context_aware:
                try:
                    chinese.append(self._generate(japanese, tuple(self.history)))
                except Exception as exc:
                    emit("error", message=str(exc))
                self.history = (self.history + [japanese])[-CONTEXT_UTTERANCES:]
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
            emit("final", utterance, japanese, "".join(chinese))
            with self.lock:
                self.active = False


def translate_only(translation_path):
    if not os.path.isdir(translation_path):
        emit("error", message="尚未完成安裝")
        return 2
    emit("status", message="準備中")
    with open(os.path.join(translation_path, "config.json")) as handle:
        model_type = json.load(handle).get("model_type", "")
    translator = Translator(*load(translation_path), model_type=model_type, name=os.path.basename(os.path.normpath(translation_path)))
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
            translator.submit(int(job.get("utterance", 0)), text)
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
    refiner_name = os.environ.get("LIVE_TRANSLATOR_FINAL_MODEL")
    if refiner_name is None:
        try:
            with open(os.path.join(models_dir, "final-model.txt")) as handle:
                refiner_name = handle.read().strip()
        except OSError:
            refiner_name = ""
    refiner_path = refiner_name if os.path.isabs(refiner_name) else os.path.join(models_dir, refiner_name)
    refiner = None
    if refiner_name and refiner_name != "none" and os.path.isdir(refiner_path):
        emit("status", message="準備中")
        refiner = Refiner(refiner_path)
    debug(f"refiner: {refiner_path if refiner else None}")
    translator = Translator(*load(translation_path), model_type=model_type, refiner=refiner, name=os.path.basename(os.path.normpath(translation_path)))
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
