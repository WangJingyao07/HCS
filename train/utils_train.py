import torch, os

def save_ckpt(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

class AverageMeter:
    def __init__(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, v, n=1):
        self.sum += float(v)*n; self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
