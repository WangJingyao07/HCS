import json, random, hashlib, os
from typing import List, Tuple

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def dedup_by_hash(records: List[dict]) -> List[dict]:
    """Remove exact duplicates by hashing the image bytes."""
    seen = set(); uniq = []
    for r in records:
        p = r["image"]
        if not os.path.isfile(p):
            uniq.append(r); continue
        h = sha256_file(p)
        if h in seen: continue
        seen.add(h); uniq.append(r)
    return uniq

def train_val_test_split(records: List[dict], ratios: Tuple[float,float,float]=(0.8,0.1,0.1), seed: int = 42):
    """Random split after duplicate removal. Returns (train, val, test)."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    records = dedup_by_hash(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    train = records[:n_tr]
    val = records[n_tr:n_tr+n_va]
    test = records[n_tr+n_va:]
    return train, val, test

def write_jsonl(path: str, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
