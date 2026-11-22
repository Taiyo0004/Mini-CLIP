"""The two encoders and the wrapper that glues them together.

Both encoders are written from scratch (no torchvision backbones, no pretrained
weights) because the point of the project was to understand the pieces.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ImageEncoder(nn.Module):
    """Small residual CNN. Roughly ResNet-18 shaped but narrower by default,
    since Flickr8k is only ~6k training images and the wide version overfit fast."""

    def __init__(self, width=48, layers=(2, 2, 2, 2), out_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        chans = [width, width * 2, width * 4, width * 8]
        blocks = []
        in_ch = width
        for stage, (ch, n) in enumerate(zip(chans, layers)):
            for b in range(n):
                stride = 2 if (b == 0 and stage > 0) else 1
                blocks.append(BasicBlock(in_ch, ch, stride))
                in_ch = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, out_dim, bias=False)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.proj(x)


class TextEncoder(nn.Module):
    """Transformer encoder over word embeddings. Features are taken from the
    <eos> position, same trick CLIP uses."""

    def __init__(self, vocab_size, context_length=32, width=256, layers=4,
                 heads=4, out_dim=512, pad_id=0, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.empty(context_length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln_final = nn.LayerNorm(width)
        self.proj = nn.Linear(width, out_dim, bias=False)

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.01)

    def forward(self, tokens):
        x = self.token_emb(tokens) + self.pos_emb[: tokens.size(1)]
        pad_mask = tokens.eq(self.pad_id)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.ln_final(x)
        # last non-pad token is the <eos>
        lengths = (~pad_mask).sum(dim=1) - 1
        x = x[torch.arange(x.size(0), device=x.device), lengths]
        return self.proj(x)


class MiniCLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, context_length=32,
                 image_width=48, text_width=256, text_layers=4, text_heads=4,
                 pad_id=0, init_temperature=0.07):
        super().__init__()
        self.visual = ImageEncoder(width=image_width, out_dim=embed_dim)
        self.textual = TextEncoder(
            vocab_size,
            context_length=context_length,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            out_dim=embed_dim,
            pad_id=pad_id,
        )
        # learned temperature, stored in log space so it stays positive
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))

    def encode_image(self, images, normalize=True):
        f = self.visual(images)
        return F.normalize(f, dim=-1) if normalize else f

    def encode_text(self, tokens, normalize=True):
        f = self.textual(tokens)
        return F.normalize(f, dim=-1) if normalize else f

    def forward(self, images, tokens):
        img = self.encode_image(images)
        txt = self.encode_text(tokens)
        # clamp keeps the temperature from collapsing, 100 is what CLIP uses
        scale = self.logit_scale.exp().clamp(max=100.0)
        return img, txt, scale
