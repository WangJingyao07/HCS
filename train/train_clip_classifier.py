
import argparse, torch, yaml
from PIL import Image

import open_clip
from torch import nn

from data_processing.transforms import build_clip_eval_transform
from train.utils_train import save_ckpt, AverageMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()


    cfg = yaml.safe_load(open(args.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(cfg['model'], pretrained=cfg['pretrained'])
    model = model.to(device).eval()


    head = nn.Linear(model.visual.output_dim, 10).to(device)  
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)



    img = build_clip_eval_transform()(Image.open(cfg['demo_image']).convert('RGB')).unsqueeze(0).to(device)
    feats = model.encode_image(img).detach()
    y = torch.tensor([0], device=device)

    criterion = nn.CrossEntropyLoss()

    meter = AverageMeter()
    for ep in range(args.epochs):
        logits = head(feats)
        loss = criterion(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        meter.update(loss.item())
        print(f"[ep {ep}] loss={meter.avg:.4f}")

    save_ckpt({'head': head.state_dict()}, 'checkpoints/clip_head.pt')

if __name__ == '__main__':
    main()
