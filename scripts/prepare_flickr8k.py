"""Builds the train/val/test split file and the vocab.

Run this once after downloading Flickr8k:
    python scripts/prepare_flickr8k.py --root data/flickr8k
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_clip.data import make_splits
from mini_clip.tokenizer import SimpleTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder with Images/ and captions.txt")
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--context-length", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    splits, captions = make_splits(args.root, seed=args.seed)

    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(args.out, "captions.json"), "w") as f:
        json.dump(captions, f)

    train_caps = [c for im in splits["train"] for c in captions[im]]
    tok = SimpleTokenizer.build(train_caps, min_freq=args.min_freq,
                                context_length=args.context_length)
    tok.save(os.path.join(args.out, "vocab.json"))

    print("images:", {k: len(v) for k, v in splits.items()})
    print("train captions:", len(train_caps))
    print("vocab size:", len(tok))


if __name__ == "__main__":
    main()
