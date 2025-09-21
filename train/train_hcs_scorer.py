
import torch, argparse
from hcs.hcs import TinyMLPScorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dim', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    model = TinyMLPScorer(in_dim=args.feat_dim).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


    for ep in range(args.epochs):
        x = torch.randn(1024, args.feat_dim).cuda()
        y = torch.randn(1024, 4).cuda()
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"[ep {ep}] loss={loss.item():.4f}")

    torch.save(model.state_dict(), 'checkpoints/hcs_scorer.pt')

if __name__ == '__main__':
    main()
