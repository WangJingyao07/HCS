import argparse, json, faiss, numpy as np
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="TRAIN split JSONL only.")
    ap.add_argument("--out", required=True, help="Output FAISS index path.")
    args = ap.parse_args()

    print("[SAFEGUARD] Use ONLY the TRAIN split here to avoid leakage.")


    feats = np.load("features.npy").astype(np.float32)
    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats)
    faiss.write_index(index, args.out)
    print(f"[OK] Wrote index to {args.out}")

if __name__ == "__main__":
    main()
