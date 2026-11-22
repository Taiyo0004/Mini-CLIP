"""Flickr8k loading. Expects the Kaggle layout: <root>/Images/*.jpg and <root>/captions.txt"""

import csv
import json
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_captions(root):
    """-> dict: image filename -> list of captions"""
    path = os.path.join(root, "captions.txt")
    caps = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # some copies of the dataset ship without a header row
        if header and header[0].strip().lower() != "image":
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            img, cap = row[0].strip(), ",".join(row[1:]).strip()
            if cap:
                caps[img].append(cap)
    return dict(caps)


def make_splits(root, seed=42, n_val=1000, n_test=1000):
    caps = read_captions(root)
    images = sorted(caps.keys())
    # only keep files that actually exist, a few are missing in some mirrors
    images = [im for im in images if os.path.exists(os.path.join(root, "Images", im))]
    rng = random.Random(seed)
    rng.shuffle(images)
    test = sorted(images[:n_test])
    val = sorted(images[n_test : n_test + n_val])
    train = sorted(images[n_test + n_val :])
    return {"train": train, "val": val, "test": test}, caps


def build_transform(image_size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Flickr8kPairs(Dataset):
    """One sample per (image, caption) pair. Used for training."""

    def __init__(self, root, images, captions, tokenizer, transform):
        self.root = root
        self.tokenizer = tokenizer
        self.transform = transform
        self.pairs = [(im, c) for im in images for c in captions[im]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        name, caption = self.pairs[i]
        img = Image.open(os.path.join(self.root, "Images", name)).convert("RGB")
        ids, _ = self.tokenizer.encode(caption)
        return self.transform(img), torch.tensor(ids, dtype=torch.long)


class ImageOnly(Dataset):
    def __init__(self, root, images, transform):
        self.root = root
        self.images = list(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, "Images", self.images[i])).convert("RGB")
        return self.transform(img)


class TextOnly(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = list(texts)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids, _ = self.tokenizer.encode(self.texts[i])
        return torch.tensor(ids, dtype=torch.long)


def flatten_eval_split(images, captions):
    """Flat caption list plus the index of the image each caption belongs to.

    Flickr8k gives 5 captions per image, so for text->image retrieval there is
    exactly one right answer, and for image->text there are 5.
    """
    texts, text2img = [], []
    for img_idx, name in enumerate(images):
        for c in captions[name]:
            texts.append(c)
            text2img.append(img_idx)
    return texts, torch.tensor(text2img, dtype=torch.long)


def load_splits(path):
    with open(path) as f:
        return json.load(f)
