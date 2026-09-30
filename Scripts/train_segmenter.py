import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

SUFFIX = 8
MAX_N = 4
BUCKETS = 1 << 18


def features(text):
    tail = text[-SUFFIX:]
    keys = [f"len{min(len(text) // 4, 10)}"]
    if text and text[-1] in "。？！?!":
        keys.append("punct")
    for n in range(1, MAX_N + 1):
        for start in range(max(0, len(tail) - n - 3), len(tail) - n + 1):
            keys.append(f"{n}:{tail[start:start + n]}:{len(tail) - start - n}")
    return keys


def bucket(key):
    value = 2166136261
    for char in key.encode("utf-8"):
        value = ((value ^ char) * 16777619) & 0xFFFFFFFF
    return value % BUCKETS


def vectorize(rows):
    from scipy.sparse import csr_matrix
    indptr = [0]
    indices = []
    for row in rows:
        cols = sorted({bucket(key) for key in features(row["text"])})
        indices.extend(cols)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return csr_matrix((data, indices, indptr), shape=(len(rows), BUCKETS))


def main():
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    train = json.load(open(source / "train.json")) + json.load(open(source / "dev.json"))
    test = json.load(open(source / "test.json"))
    model = LogisticRegression(C=2.0, max_iter=2000)
    model.fit(vectorize(train), [row["label"] for row in train])
    accuracy = model.score(vectorize(test), [row["label"] for row in test])
    print(f"train rows={len(train)} test rows={len(test)} test accuracy={accuracy:.4f}")
    weights = model.coef_[0]
    nonzero = np.nonzero(weights)[0]
    target.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"suffix": SUFFIX, "max_n": MAX_N, "buckets": BUCKETS, "bias": float(model.intercept_[0]), "weights": {int(index): round(float(weights[index]), 5) for index in nonzero}},
        open(target / "weights.json", "w"),
    )
    print(f"saved {len(nonzero)} weights to {target / 'weights.json'}")


if __name__ == "__main__":
    main()
