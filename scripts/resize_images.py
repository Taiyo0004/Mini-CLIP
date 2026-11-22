"""Pre-resize Flickr8k once so the dataloader is not the bottleneck.

Decoding full-size JPEGs every epoch pegged my CPU while the GPU sat idle. This
writes a copy with the short side at --size, which is about 4x faster to load.

    python scripts/resize_images.py --root data/flickr8k --out data/flickr8k-160 --size 160
"""

import argparse
import os
import shutil

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=160, help="target short side")
    p.add_argument("--quality", type=int, default=92)
    args = p.parse_args()

    src_dir = os.path.join(args.root, "Images")
    dst_dir = os.path.join(args.out, "Images")
    os.makedirs(dst_dir, exist_ok=True)

    names = [n for n in sorted(os.listdir(src_dir)) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    for i, name in enumerate(names):
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        img = Image.open(os.path.join(src_dir, name)).convert("RGB")
        w, h = img.size
        scale = args.size / min(w, h)
        if scale < 1:
            img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        img.save(dst, quality=args.quality)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(names)}")

    # captions.txt has to travel with the images, the loaders expect it next to Images/
    shutil.copy(os.path.join(args.root, "captions.txt"), os.path.join(args.out, "captions.txt"))
    print(f"wrote {len(names)} images to {dst_dir}")


if __name__ == "__main__":
    main()
