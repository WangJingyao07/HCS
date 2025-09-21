import open_clip, torch

class OpenCLIPEncoder:
    """
    Thin wrapper around OpenCLIP for a consistent encode_image/encode_text API.
    """
    def __init__(self, arch="ViT-B-32", ckpt="laion2b_s34b_b79k", device="cuda"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=ckpt)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode_image(self, x):
        return self.model.encode_image(x)

    @torch.no_grad()
    def encode_text(self, t):
        return self.model.encode_text(t)

    @property
    def visual_dim(self):
        return self.model.visual.output_dim
