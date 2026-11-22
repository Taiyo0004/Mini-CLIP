"""Train Mini CLIP on Flickr8k.

    python scripts/train.py --root data/flickr8k --epochs 30 --batch-size 128
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_clip.data import Flickr8kPairs, build_transform, load_splits
from mini_clip.engine import cosine_lr, evaluate, train_one_epoch
from mini_clip.model import MiniCLIP
from mini_clip.tokenizer import SimpleTokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--data", default="data", help="where prepare_flickr8k.py wrote its files")
    p.add_argument("--out", default="runs/mini-clip")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200, help="warmup steps")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--eval-every", type=int, default=1, help="run val every N epochs")
    p.add_argument("--channels-last", action="store_true",
                   help="slightly faster convs on tensor-core GPUs")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--image-width", type=int, default=48)
    p.add_argument("--text-width", type=int, default=256)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--text-heads", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=0,
                   help="keep at 0 on Windows, spawning workers per eval can deadlock")
    p.add_argument("--resume", default=None, help="path to a checkpoint to continue from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="mixed precision, cuda only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    splits = load_splits(os.path.join(args.data, "splits.json"))
    with open(os.path.join(args.data, "captions.json")) as f:
        captions = json.load(f)
    tokenizer = SimpleTokenizer.load(os.path.join(args.data, "vocab.json"))

    train_ds = Flickr8kPairs(
        args.root, splits["train"], captions, tokenizer,
        build_transform(args.image_size, train=True),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=args.workers > 0,
    )
    eval_tf = build_transform(args.image_size, train=False)

    model = MiniCLIP(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        context_length=tokenizer.context_length,
        image_width=args.image_width,
        text_width=args.text_width,
        text_layers=args.text_layers,
        text_heads=args.text_heads,
        pad_id=tokenizer.pad_id,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.1f}M params, {len(train_ds)} training pairs")

    # no weight decay on norms, biases or the temperature
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "logit_scale" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
    )

    total_steps = args.epochs * len(train_loader)
    set_lr = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    start_epoch, best, step = 1, -1.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        step = ck.get("step", (start_epoch - 1) * len(train_loader))
        best = ck.get("best", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(args.out, "log.csv")
    mode = "a" if args.resume and os.path.exists(log_path) else "w"
    with open(log_path, mode, newline="") as f:
        if mode == "w":
            csv.writer(f).writerow(
                ["epoch", "train_loss", "t2i_R@1", "t2i_R@5", "t2i_R@10",
                 "i2t_R@1", "i2t_R@5", "i2t_R@10", "rsum", "minutes"]
            )

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        loss, step = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler,
            set_lr=set_lr, start_step=step, channels_last=args.channels_last,
        )
        last_epoch = epoch == args.epochs
        if epoch % args.eval_every and not last_epoch:
            print(f"epoch {epoch:3d}  loss {loss:.3f}  (val skipped)")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "args": vars(args), "vocab_size": len(tokenizer),
                        "epoch": epoch, "step": step, "best": best, "val": None},
                       os.path.join(args.out, "last.pt"))
            continue

        m = evaluate(model, args.root, splits["val"], captions, tokenizer,
                     eval_tf, device, num_workers=args.eval_workers)
        mins = (time.time() - t0) / 60
        print(f"epoch {epoch:3d}  loss {loss:.3f}  "
              f"t2i R@1 {m['t2i_R@1']:.1f} R@5 {m['t2i_R@5']:.1f}  "
              f"i2t R@5 {m['i2t_R@5']:.1f}  rsum {m['rsum']:.1f}  [{mins:.1f} min]")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(loss, 4),
                *[round(m[k], 2) for k in
                  ("t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10", "rsum")],
                round(mins, 1),
            ])

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "vocab_size": len(tokenizer),
            "epoch": epoch,
            "step": step,
            "best": max(best, m["rsum"]),
            "val": m,
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if m["rsum"] > best:
            best = m["rsum"]
            torch.save(ckpt, os.path.join(args.out, "best.pt"))

    print(f"done, best val rsum {best:.1f} -> {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
