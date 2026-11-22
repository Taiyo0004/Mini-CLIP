import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ImageOnly, TextOnly, flatten_eval_split
from .loss import clip_loss
from .metrics import retrieval_metrics


def cosine_lr(optimizer, base_lr, warmup_steps, total_steps):
    def set_lr(step):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / max(1, warmup_steps)
        else:
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = 0.5 * base_lr * (1 + math.cos(math.pi * min(t, 1.0)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr
    return set_lr


def train_one_epoch(model, loader, optimizer, device, scaler=None,
                    set_lr=None, start_step=0, grad_clip=1.0, log_every=50):
    model.train()
    running, seen = 0.0, 0
    step = start_step
    bar = tqdm(loader, desc="train", leave=False)
    for images, tokens in bar:
        if set_lr is not None:
            set_lr(step)
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, enabled=amp):
            img_f, txt_f, scale = model(images, tokens)
            loss = clip_loss(img_f, txt_f, scale)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(100.0))

        running += loss.item() * images.size(0)
        seen += images.size(0)
        step += 1
        if step % log_every == 0:
            bar.set_postfix(loss=f"{running / max(1, seen):.3f}")

    return running / max(1, seen), step


@torch.no_grad()
def encode_split(model, root, images, captions, tokenizer, transform, device,
                 batch_size=64, num_workers=2):
    """Embed every image and every caption in a split."""
    model.eval()
    texts, text2img = flatten_eval_split(images, captions)

    img_loader = DataLoader(ImageOnly(root, images, transform), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    txt_loader = DataLoader(TextOnly(texts, tokenizer), batch_size=batch_size * 4,
                            shuffle=False, num_workers=num_workers)

    img_feats = []
    for batch in tqdm(img_loader, desc="images", leave=False):
        img_feats.append(model.encode_image(batch.to(device)).float().cpu())
    txt_feats = []
    for batch in tqdm(txt_loader, desc="captions", leave=False):
        txt_feats.append(model.encode_text(batch.to(device)).float().cpu())

    return torch.cat(img_feats), torch.cat(txt_feats), text2img


def evaluate(model, root, images, captions, tokenizer, transform, device, **kw):
    img_f, txt_f, text2img = encode_split(
        model, root, images, captions, tokenizer, transform, device, **kw
    )
    return retrieval_metrics(img_f, txt_f, text2img)
