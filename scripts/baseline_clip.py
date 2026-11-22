"""Pretrained CLIP (ViT-B/32) on the same test split, for reference.

Needs `pip install transformers`. This is the number my model gets compared
against in the README, so it has to use the same images, the same captions and
the same metric code.

    python scripts/baseline_clip.py --root data/flickr8k
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_clip.data import flatten_eval_split, load_splits
from mini_clip.metrics import retrieval_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    splits = load_splits(os.path.join(args.data, "splits.json"))
    with open(os.path.join(args.data, "captions.json")) as f:
        captions = json.load(f)
    images = splits[args.split]
    texts, text2img = flatten_eval_split(images, captions)

    img_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), args.batch_size), desc="images"):
            batch = [
                Image.open(os.path.join(args.root, "Images", n)).convert("RGB")
                for n in images[i : i + args.batch_size]
            ]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            f = model.get_image_features(**inputs)
            img_feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())

        txt_feats = []
        for i in tqdm(range(0, len(texts), args.batch_size * 4), desc="captions"):
            inputs = proc(text=texts[i : i + args.batch_size * 4], return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(device)
            f = model.get_text_features(**inputs)
            txt_feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())

    m = retrieval_metrics(torch.cat(img_feats), torch.cat(txt_feats), text2img)
    print(f"\n{args.model} zero-shot, {args.split} split ({len(images)} images)")
    for d in ("t2i", "i2t"):
        print(f"  {d}: " + "  ".join(f"R@{k} {m[f'{d}_R@{k}']:.1f}" for k in (1, 5, 10)))
    print(f"  rsum: {m['rsum']:.1f}")


if __name__ == "__main__":
    main()
