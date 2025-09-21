import argparse, json, os, hashlib

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_items(jsonl_path: str):
    items = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")]
    return items

def check_overlap(train_jsonl: str, val_jsonl: str, test_jsonl: str):
    def hash_set(items):
        out = set()
        for it in items:
            p = it["image"]
            if os.path.isfile(p):
                out.add(sha256_file(p))
        return out
    tr, va, te = map(load_items, [train_jsonl, val_jsonl, test_jsonl])
    Htr, Hva, Hte = hash_set(tr), hash_set(va), hash_set(te)
    return (Htr & Hva, Htr & Hte, Hva & Hte)

def check_label_leak_in_filenames(items, label_key="label"):
    warnings = []
    for it in items:
        p = os.path.basename(it["image"]).lower()
        lbl = str(it.get(label_key, "")).lower()
        if lbl and lbl in p:
            warnings.append(it["image"])
    return warnings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    tr = load_items(args.train)
    va = load_items(args.val)
    te = load_items(args.test)

    ov_tr_va, ov_tr_te, ov_va_te = check_overlap(args.train, args.val, args.test)
    if ov_tr_va or ov_tr_te or ov_va_te:
        print("[LEAK] Hash overlaps detected:")
        if ov_tr_va: print(f"  train ∩ val: {len(ov_tr_va)}")
        if ov_tr_te: print(f"  train ∩ test: {len(ov_tr_te)}")
        if ov_va_te: print(f"  val ∩ test: {len(ov_va_te)}")
    else:
        print("[OK] No hash overlaps across splits.")

    for name, items in [("train", tr), ("val", va), ("test", te)]:
        warns = check_label_leak_in_filenames(items)
        if warns:
            print(f"[WARN] {name}: {len(warns)} files whose basename contains the label string (avoid exposing paths).")

    print("[INFO] Transforms use fixed CLIP stats and are split-safe (see data_processing/transforms.py).")

if __name__ == "__main__":
    main()
