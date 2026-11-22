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

These are the exact commands behind the numbers in the results table:

```bash
# resize once, the dataloader is the bottleneck otherwise
python scripts/resize_images.py --root data/flickr8k --out data/flickr8k-160 --size 160
python scripts/prepare_flickr8k.py --root data/flickr8k-160

python scripts/train.py --root data/flickr8k-160 \
    --image-size 128 --batch-size 128 --amp --channels-last \
    --workers 4 --eval-workers 0 --epochs 40 --eval-every 2

python scripts/evaluate.py --root data/flickr8k-160 --ckpt runs/mini-clip/best.pt --examples 3

# baseline reads the originals, that is the resolution its preprocessing expects
python scripts/baseline_clip.py --root data/flickr8k --batch-size 32
```

Sanity checks that need no dataset and take a couple of seconds:

```bash
python tests/test_shapes.py
```

### On a 4 GB GPU

This is what I train on (RTX 2050 laptop, 4 GB). The model is small, so VRAM is
not really the constraint, the CPU is: 30k JPEG decodes per epoch will starve the
GPU long before you run out of memory. So resize once first.

```bash
python scripts/resize_images.py --root data/flickr8k --out data/flickr8k-160 --size 160
python scripts/prepare_flickr8k.py --root data/flickr8k-160

python scripts/train.py --root data/flickr8k-160 \
    --image-size 128 --batch-size 128 --amp --channels-last \
    --workers 4 --epochs 40 --eval-every 2
```

Rough VRAM at batch 128 with AMP: around 1.5 GB at 128px and under 2.5 GB at
224px, so either resolution fits, but 128px is about 3x faster per step and on a
6k-image dataset the extra resolution bought me very little. If you do hit OOM,
drop to `--batch-size 64 --lr 2e-4` before touching anything else, and close
anything using the GPU for display.

On Windows, leave `--eval-workers` at 0. Every DataLoader spawns fresh processes
there rather than forking, and creating two of them per validation pass, twenty
times in a run, deadlocked on me around epoch 28. Validation is only 1,000 images
so single-process decoding costs a few seconds. If a run does die, `last.pt`
carries the optimizer state, so `--resume runs/mini-clip/last.pt` picks it up at
the next epoch.

Keep the batch size as high as it will go. Every other pair in the batch is a
negative, so shrinking the batch directly weakens the training signal, which is
why I would rather cut resolution than batch size.

The pretrained baseline runs inference only and fits fine in 4 GB:

```bash
python scripts/baseline_clip.py --root data/flickr8k-160 --batch-size 32
```

Split is a seeded shuffle of the 8,091 images into 6,091 train / 1,000 val / 1,000 test,
written to `data/splits.json` so every run and the baseline see the same test images.
Model selection is on val rsum, test numbers are reported once at the end.

## Results

Flickr8k test split, 1,000 images and 5,000 captions. t2i is caption to image
(one correct image), i2t is image to caption (5 correct captions, best rank counts).

| model | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |
|---|---|---|---|---|---|---|
| Mini CLIP (from scratch, 6k images) | 4.6 | 15.8 | 24.3 | 5.2 | 17.0 | 25.9 |
| CLIP ViT-B/32, zero-shot | 55.0 | 80.8 | 87.9 | 72.0 | 91.1 | 95.8 |
| chance | 0.1 | 0.5 | 1.0 | 0.1 | 0.5 | 1.0 |

Median rank is 48 for t2i and 47 for i2t out of 1,000 candidates. rsum is 92.7
for Mini CLIP against 482.6 for the pretrained baseline.

Training run behind those numbers: 40 epochs at 128px, batch 128, AdamW at 3e-4
with warmup and cosine decay, mixed precision, about 40 minutes on an RTX 2050
(4 GB). Best validation rsum was at epoch 38. Per-epoch curves are in
`runs/mini-clip/log.csv`.

R@5 of 15.8 is about 30x chance, so the two encoders did learn a shared space,
and it is also 5x below pretrained CLIP on the same 1,000 images scored by the
same code. That gap is the most useful thing the project produced. The objective
is the same, the loss is the same, the evaluation is the same. What differs is
10.9M parameters trained on 30,455 caption pairs against roughly 151M trained on
400M pairs, which is about four orders of magnitude more supervision. Contrastive
alignment is cheap to implement and extremely expensive to make work, and the
table is what that sentence looks like as numbers.

The baseline is run on the original full resolution images, which is the setting
CLIP's preprocessing expects, so it is not handicapped by my 160px copies.

## Notes from building it

- Overfitting is the whole story on a dataset this size. Training loss falls from
  4.51 to 0.13 over 40 epochs while validation R@5 flattens out around epoch 14 and
  then moves by less than two points. Narrowing the image encoder to 48 channels at
  the stem instead of 64, plus colour jitter and random resized crop, slowed that
  down but nowhere near stopped it. More data is the only real fix.
- Because the batch supplies the negatives, batch size is a training signal and not
  just a throughput knob: at batch 128 each positive pair is scored against 127
  negatives. On 4 GB that is the reason to cut resolution rather than batch size.
- The learned temperature is clamped at exp(4.6) ~ 100 as in the paper, and the
  text feature is read from the `<eos>` position rather than mean pooled, also
  following the paper. I have not ablated either choice yet.
- Dropping words that appear once keeps the vocabulary small, which helps on 30k
  captions but means rare words all collapse to `<unk>`. BPE is the obvious upgrade.
- Retrieval failures are mostly near misses rather than nonsense. A caption about a
  brown and white dog in snow put the right image second, and the four wrong ones
  were also dogs in snow. The model has the scene category and loses the instance.

## Things I would do next

- Ablate the things listed above that I took from the paper on faith: `<eos>` vs
  mean pooling, and batch 32 / 64 / 128 at fixed epochs.
- Swap the CNN for a small ViT and compare at matched parameter count.
- Train on a larger caption set (COCO, or Flickr30k) to see where the curve bends.
- Hard negative mining inside the batch.
- Try initialising the text encoder from a pretrained one and keeping the image side
  from scratch, to see which side is the bottleneck.

## Reference

Radford et al., *Learning Transferable Visual Models From Natural Language
Supervision*, 2021.
