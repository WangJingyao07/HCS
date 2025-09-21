from torch.utils.data import Dataset
from PIL import Image
import json, os
from typing import Optional

class VQADataset(Dataset):
    """Label-agnostic transforms; keep file paths away from the model."""
    def __init__(self, ann_file: str, transform=None, root: Optional[str]=None):
        self.items = [json.loads(line) for line in open(ann_file, "r", encoding="utf-8")]
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = rec["image"]
        if self.root is not None and not os.path.isabs(img_path):
            img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "question": rec["question"], "answer": rec.get("answer", None)}
