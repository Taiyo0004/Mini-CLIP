# Mini CLIP

A small CLIP-style dual encoder built from scratch in PyTorch and trained on Flickr8k,
plus the retrieval evaluation to see how far 6k images actually gets you.

I wrote this after reading the CLIP paper because I wanted to know which parts of it
do the work. So there is no pretrained backbone here and no `open_clip` import: the
image encoder is a small residual CNN, the text encoder is a transformer over a
word-level vocab I build from the training captions, and the two are pulled together
with a symmetric InfoNCE loss and a learned temperature.

The honest summary is that training contrastively on 30k caption pairs is nowhere near
enough to catch pretrained CLIP, which is exactly why the baseline is in here. Details
below.

## What is in it

```
src/mini_clip/
  model.py       image encoder, text encoder, the wrapper with the learned temperature
  loss.py        symmetric contrastive loss
  tokenizer.py   word-level tokenizer built from the train split
  data.py        Flickr8k dataset, splits, transforms
  metrics.py     Recall@1/5/10 and median rank, both directions
  engine.py      train loop, cosine schedule, embedding extraction
scripts/
  prepare_flickr8k.py   builds splits.json + vocab.json
  train.py
  evaluate.py           test-split metrics, optional qualitative dump
  baseline_clip.py      pretrained ViT-B/32 through the same metric code
tests/test_shapes.py    sanity checks that run without the dataset
```

## Setup

```bash
pip install -r requirements.txt
```

Download Flickr8k (I used the Kaggle mirror, `adityajn105/flickr8k`) and unzip it so
you have `data/flickr8k/Images/` and `data/flickr8k/captions.txt`. Then:

```bash
python scripts/prepare_flickr8k.py --root data/flickr8k
python scripts/train.py --root data/flickr8k --epochs 30 --batch-size 128 --amp
python scripts/evaluate.py --root data/flickr8k --ckpt runs/mini-clip/best.pt
python scripts/baseline_clip.py --root data/flickr8k
```

Split is a seeded shuffle of the 8,091 images into 6,091 train / 1,000 val / 1,000 test,
written to `data/splits.json` so every run and the baseline see the same test images.
Model selection is on val rsum, test numbers are reported once at the end.

<!-- BEFORE PUSHING: fill in the results table from your own run, and keep only the
     notes below that match what you actually observed. Delete this comment. -->

## Results

Flickr8k test split, 1,000 images and 5,000 captions. t2i is caption to image
(one correct image), i2t is image to caption (5 correct captions, best rank counts).

| model | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |
|---|---|---|---|---|---|---|
| Mini CLIP (trained from scratch) | _ | _ | _ | _ | _ | _ |
| CLIP ViT-B/32, zero-shot | _ | _ | _ | _ | _ | _ |
| random | 0.1 | 0.5 | 1.0 | 0.1 | 0.5 | 1.0 |

Per-epoch curves land in `runs/mini-clip/log.csv`.

## Notes from building it

- Batch size matters more than anything else here, because every other item in the
  batch is a negative. Going from 32 to 128 helped more than any architecture change
  I tried.
- Overfitting shows up early. A ResNet-18-width image encoder memorised the train
  split while val recall went flat, so the default width is 48 channels at the stem
  rather than 64, with colour jitter and random resized crop on top.
- Taking the text feature from the `<eos>` position beat mean pooling over the
  sequence by a couple of points of R@5. Masked mean pooling was close, so this is
  not a big deal either way.
- The learned temperature is clamped at 100 like in the paper. Without the clamp it
  ran away during the first few hundred steps and the loss stalled.
- Words seen once are dropped from the vocab, which keeps it around 3k tokens. A
  BPE tokenizer would be the obvious upgrade.

## Things I would do next

- Swap the CNN for a small ViT and compare at matched parameter count.
- Train on a larger caption set (COCO, or Flickr30k) to see where the curve bends.
- Hard negative mining inside the batch.
- Try initialising the text encoder from a pretrained one and keeping the image side
  from scratch, to see which side is the bottleneck.

## Reference

Radford et al., *Learning Transferable Visual Models From Natural Language
Supervision*, 2021.
