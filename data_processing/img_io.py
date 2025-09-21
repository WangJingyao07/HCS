from PIL import Image
import torch
import numpy as np

def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Convert a single image tensor [1,3,H,W] in [0,1] to PIL.Image.
    """
    assert img.ndim == 4 and img.shape[0] == 1, "Expect [1,3,H,W]"
    x = img[0].detach().cpu().clamp(0,1)
    x = (x * 255.0).byte().permute(1,2,0).numpy()
    return Image.fromarray(x)

def apply_mask_pil(pil_img: Image.Image, mask: torch.Tensor) -> Image.Image:
    """
    Apply a binary mask [1,1,H,W] to a PIL image. Masked-out pixels set to black.
    """
    arr = np.array(pil_img).astype("float32") / 255.0  # [H,W,3]
    c, h, w = 3, arr.shape[0], arr.shape[1]
    img = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    assert mask.ndim == 4 and mask.shape[1] == 1, "Expect [1,1,H,W]"
    img = (img * mask.clamp(0,1)).clamp(0,1)
    return tensor_to_pil(img)
