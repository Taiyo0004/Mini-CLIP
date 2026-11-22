"""Evaluate a trained checkpoint on the test split.

    python scripts/evaluate.py --root data/flickr8k --ckpt runs/mini-clip/best.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_clip.data import build_transform, flatten_eval_split, load_splits
from mini_clip.engine import encode_split
from mini_clip.metrics import retrieval_metrics
from mini_clip.model import MiniCLIP
from mini_clip.tokenizer import SimpleTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--examples", type=int, default=0,
                   help="print this many caption -> top-5 image retrievals")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved = ckpt["args"]

    splits = load_splits(os.path.join(args.data, "splits.json"))
    with open(os.path.join(args.data, "captions.json")) as f:
        captions = json.load(f)
    tokenizer = SimpleTokenizer.load(os.path.join(args.data, "vocab.json"))

    model = MiniCLIP(
        vocab_size=ckpt["vocab_size"],
        embed_dim=saved["embed_dim"],
        context_length=tokenizer.context_length,
        image_width=saved["image_width"],
        text_width=saved["text_width"],
        text_layers=saved["text_layers"],
        text_heads=saved["text_heads"],
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    images = splits[args.split]
    tf = build_transform(saved["image_size"], train=False)
    img_f, txt_f, text2img = encode_split(
        model, args.root, images, captions, tokenizer, tf, device, num_workers=args.workers
    )
    m = retrieval_metrics(img_f, txt_f, text2img)

    print(f"\n{args.split} split: {len(images)} images, {txt_f.size(0)} captions "
          f"(checkpoint from epoch {ckpt['epoch']})")
    for direction in ("t2i", "i2t"):
        print(f"  {direction}: " + "  ".join(
            f"R@{k} {m[f'{direction}_R@{k}']:.1f}" for k in (1, 5, 10)
        ) + f"  medR {m[f'{direction}_medr']:.0f}")
    print(f"  rsum: {m['rsum']:.1f}")

    if args.examples:
        texts, _ = flatten_eval_split(images, captions)
        sims = txt_f @ img_f.t()
        top = sims.topk(5, dim=1).indices
        print("\nqualitative:")
        for i in range(0, min(args.examples * 5, len(texts)), 5):
            print(f'  "{texts[i]}"')
            print(f"    gold: {images[int(text2img[i])]}")
            print("    top5: " + ", ".join(images[j] for j in top[i].tolist()))


if __name__ == "__main__":
    main()
