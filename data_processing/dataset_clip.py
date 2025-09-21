from typing import List, Optional
from PIL import Image
from torch.utils.data import Dataset
import json, os

class CLIPImageDataset(Dataset):
    """Image classification dataset for CLIP-style usage (no label-dependent transforms)."""
    def __init__(self, ann_file: str, label_list: List[str],
                 transform=None, allow_labels: bool=True, root: Optional[str]=None):
        self.items = [json.loads(line) for line in open(ann_file, "r", encoding="utf-8")]
        self.label_to_id = {l:i for i,l in enumerate(label_list)}
        self.transform = transform
        self.allow_labels = allow_labels
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
        y = self.label_to_id.get(rec.get("label",""), -1) if self.allow_labels else -1
        return img, y
