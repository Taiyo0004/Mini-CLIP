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


def image_embeds(model, pixel_values):
    """Vision tower -> pooled -> joint space.

    Not using get_image_features: what it returns has changed between
    transformers versions (tensor in some, an output object in others, and in at
    least one the pooled vector is already projected), and for ViT-B/32 you
    cannot tell projected from unprojected by shape because the text tower is
    512 wide and so is the joint space. Calling the submodules is unambiguous.
    """
    pooled = model.vision_model(pixel_values=pixel_values).pooler_output
    return model.visual_projection(pooled)


def text_embeds(model, input_ids, attention_mask=None):
    pooled = model.text_model(input_ids=input_ids,
                              attention_mask=attention_mask).pooler_output
    return model.text_projection(pooled)


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
            f = image_embeds(model, inputs["pixel_values"])
            assert f.shape[-1] == model.config.projection_dim, f.shape
            img_feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())

        txt_feats = []
        for i in tqdm(range(0, len(texts), args.batch_size * 4), desc="captions"):
            inputs = proc(text=texts[i : i + args.batch_size * 4], return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(device)
            f = text_embeds(model, inputs["input_ids"], inputs.get("attention_mask"))
            assert f.shape[-1] == model.config.projection_dim, f.shape
            txt_feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())

    img_f, txt_f = torch.cat(img_feats), torch.cat(txt_feats)

    # cheap wiring check: matched pairs should sit well above random ones. If the
    # projections were applied wrongly this comes out near zero and the recall
    # numbers below would be quietly meaningless instead of obviously broken.
    matched = (txt_f * img_f[text2img]).sum(-1).mean().item()
    shuffled = (txt_f * img_f[text2img[torch.randperm(txt_f.size(0))]]).sum(-1).mean().item()
    print(f"\nmean cosine, matched {matched:.3f} vs shuffled {shuffled:.3f}")
    if matched - shuffled < 0.05:
        print("  that gap is too small, the features are probably wired wrong")

    m = retrieval_metrics(img_f, txt_f, text2img)
    print(f"\n{args.model} zero-shot, {args.split} split ({len(images)} images)")
    for d in ("t2i", "i2t"):
        print(f"  {d}: " + "  ".join(f"R@{k} {m[f'{d}_R@{k}']:.1f}" for k in (1, 5, 10)))
    print(f"  rsum: {m['rsum']:.1f}")


if __name__ == "__main__":
    main()
