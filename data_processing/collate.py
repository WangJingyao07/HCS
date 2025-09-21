import torch

def collate_simple(batch):
    """Collate for classification or VQA without exposing file paths."""
    if isinstance(batch[0], tuple):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return imgs, ys
    else:
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        qs = [b["question"] for b in batch]
        ans = [b.get("answer") for b in batch]
        return {"image": imgs, "question": qs, "answer": ans}
