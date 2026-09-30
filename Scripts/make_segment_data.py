import json
import random
import re
import sys
from pathlib import Path

CUT_AFTER = re.compile(r"(、|て|で|が|と|ので|けど|から|は|を|に|の|し|そして|また|それから|って|たり|へ|も|や|まで|なら|ば|ながら)")
TERMINAL = "。？！?!"


def cuts(sentence, rng):
    body = sentence.rstrip(TERMINAL)
    out = set()
    for match in CUT_AFTER.finditer(body):
        end = match.end()
        if 3 <= end <= len(body) - 2:
            out.add(body[:end])
    for _ in range(2):
        end = rng.randint(3, max(3, len(body) - 2))
        if end <= len(body) - 2:
            out.add(body[:end])
    out.discard(body)
    return sorted(out)


def load_sentences(path):
    seen = set()
    for document in json.load(open(path)):
        for turn in document["conversation"]:
            text = turn["ja_sentence"].strip()
            if 4 <= len(text) <= 80 and text not in seen:
                seen.add(text)
                yield text


def build(sentences, rng, limit_cuts=3):
    rows = []
    sentences = list(sentences)
    for index, sentence in enumerate(sentences):
        body = sentence.rstrip(TERMINAL)
        rows.append({"text": sentence, "label": 1})
        rows.append({"text": body, "label": 1})
        if index + 1 < len(sentences) and rng.random() < 0.3:
            rows.append({"text": body + "、" + sentences[index + 1].rstrip(TERMINAL), "label": 1})
        pieces = cuts(sentence, rng)
        rng.shuffle(pieces)
        for piece in pieces[:limit_cuts]:
            rows.append({"text": piece, "label": 0})
        if index + 1 < len(sentences) and rng.random() < 0.3 and pieces:
            rows.append({"text": body + "。" + pieces[0], "label": 0})
    unique = {}
    for row in rows:
        unique.setdefault(row["text"], row["label"])
    rows = [{"text": text, "label": label} for text, label in unique.items()]
    rng.shuffle(rows)
    return rows


def write_laya(rows, path):
    with open(path, "w") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({"id": f"seg-{index}", "state": {"utterance": row["text"]}, "gold": {"complete": bool(row["label"])}}, ensure_ascii=False) + "\n")


def fetch(source):
    import urllib.request
    source.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        path = source / f"{split}.json"
        if not path.exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/tsuruoka-lab/BSD/master/{split}.json", path)


def load_extra(directory):
    if directory is None or not directory.is_dir():
        return []
    sentences = []
    for path in sorted(directory.glob("*.txt")):
        for line in open(path, encoding="utf-8"):
            text = line.strip()
            if 4 <= len(text) <= 120:
                sentences.append(text)
    return sentences


def main():
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    extra = load_extra(Path(sys.argv[3]) if len(sys.argv) > 3 else target / "extra")
    fetch(source)
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    if extra:
        print(f"extra sentences: {len(extra)} (weighted x3 into train)")
    for split in ("train", "dev", "test"):
        sentences = list(load_sentences(source / f"{split}.json"))
        if split == "train":
            sentences += extra * 3
        rows = build(sentences, rng)
        json.dump(rows, open(target / f"{split}.json", "w"), ensure_ascii=False)
        write_laya(rows, target / f"laya_{split}.jsonl")
        positives = sum(row["label"] for row in rows)
        print(f"{split}: {len(rows)} rows, complete={positives}, cut={len(rows) - positives}")


if __name__ == "__main__":
    main()
