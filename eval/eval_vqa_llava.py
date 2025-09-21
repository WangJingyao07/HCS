import argparse, yaml, torch
from PIL import Image
from data_processing.transforms import build_clip_eval_transform
from data_processing.img_io import apply_mask_pil
from hcs.hcs import HCS, HCSWeights
from models.llava.llava_wrapper import LlavaWrapper
import open_clip  # for HCS features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Overrides config.model_path if given")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_path = args.model_path or cfg["model_path"]

    image_path = cfg.get("image", "data/sample_images/example.jpg")
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = build_clip_eval_transform(image_size=224)(img_pil).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = img_tensor.to(device)

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model = clip_model.to(device).eval()

    def dummy_head(feat):
        with torch.no_grad():
            anchors = torch.randn(5, feat.shape[-1], device=feat.device)
            anchors = torch.nn.functional.normalize(anchors, dim=-1)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            return feat @ anchors.t()

    if cfg.get("hcs", {}).get("use", True):
        grids = [tuple(g) for g in cfg["hcs"]["grids"]]
        topks = cfg["hcs"]["topks"]
        hcs = HCS(clip_model.encode_image, dummy_head, feat_dim=clip_model.visual.output_dim, weights=HCSWeights())
        out = hcs.hierarchical_select(img_tensor, levels=grids, topk_per_level=topks)
        masked_img_pil = apply_mask_pil(img_pil, out["mask"])
    else:
        masked_img_pil = img_pil

    q = cfg.get("question", "Describe the scene.")
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    temperature = float(cfg.get("temperature", 0.2))

    llava = LlavaWrapper(model_path=model_path, dtype="bfloat16")
    ans = llava.generate(masked_img_pil, q, max_new_tokens=max_new_tokens, temperature=temperature)
    print("[Question]", q)
    print("[Answer]", ans)

if __name__ == "__main__":
    main()
