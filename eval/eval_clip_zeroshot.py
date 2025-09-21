import argparse, yaml, torch
from PIL import Image
import open_clip
from data_processing.transforms import build_clip_eval_transform
from hcs.hcs import HCS, HCSWeights

def build_clip_head(model, device):

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    labels = ["airport", "beach", "bridge", "farmland", "forest"]
    text = tokenizer(labels).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

    def head(feat):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat @ text_feat.t()
    return head, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--use_hcs", type=str, default="true")
    parser.add_argument("--hcs_grid", type=str, default="[(4,4),(8,8)]")
    parser.add_argument("--hcs_topk", type=str, default="[6,8]")
    args = parser.parse_args()


    # parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--use_hcs", type=str, default="true")
    # parser.add_argument("--hcs_grid", type=str, default="[(4,4),(8,8)]")
    # parser.add_argument("--hcs_topk", type=str, default="[6,8]")
    # args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(cfg["model"], pretrained=cfg["pretrained"])    
    model.to(device).eval()
    head, labels = build_clip_head(model, device)

    img = build_clip_eval_transform()(Image.open(cfg["demo_image"]).convert("RGB")).unsqueeze(0).to(device)

    if args.use_hcs.lower() == "true":
        grids = eval(args.hcs_grid)
        topks = eval(args.hcs_topk)
        hcs = HCS(model.encode_image, head, feat_dim=model.visual.output_dim, weights=HCSWeights())
        out = hcs.hierarchical_select(img, levels=grids, topk_per_level=topks)
        logits = hcs.fuse_and_forward(img, out["mask"])
    else:
        with torch.no_grad():
            feats = model.encode_image(img)
            logits = head(feats)

    pred = logits.argmax(dim=-1).item()
    print("Prediction:", labels[pred])

if __name__ == "__main__":
    main()
