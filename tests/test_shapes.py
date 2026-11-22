"""Quick sanity checks that do not need the dataset.

    pytest -q    (or just: python tests/test_shapes.py)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_clip.loss import clip_loss
from mini_clip.metrics import retrieval_metrics
from mini_clip.model import MiniCLIP
from mini_clip.tokenizer import SimpleTokenizer

CAPTIONS = [
    "a dog runs across a grassy field",
    "two children play with a red ball",
    "a man in a blue shirt climbs a rock",
    "a dog swims in a lake",
]


def build():
    tok = SimpleTokenizer.build(CAPTIONS, min_freq=1, context_length=16)
    model = MiniCLIP(len(tok), embed_dim=64, context_length=16, image_width=8,
                     text_width=32, text_layers=2, text_heads=2, pad_id=tok.pad_id)
    return tok, model


def test_tokenizer_roundtrip():
    tok, _ = build()
    ids, length = tok.encode(CAPTIONS[0])
    assert len(ids) == 16
    assert ids[0] == tok.bos_id and ids[length - 1] == tok.eos_id
    assert tok.decode(ids) == CAPTIONS[0]
    # unknown words fall back to <unk>
    ids2, _ = tok.encode("a zzzzz flies")
    assert tok.unk_id in ids2


def test_forward_shapes_and_norm():
    tok, model = build()
    images = torch.randn(4, 3, 64, 64)
    tokens = torch.tensor([tok.encode(c)[0] for c in CAPTIONS])
    img_f, txt_f, scale = model(images, tokens)
    assert img_f.shape == (4, 64) and txt_f.shape == (4, 64)
    assert torch.allclose(img_f.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(txt_f.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert scale > 0


def test_loss_is_lower_for_aligned_features():
    torch.manual_seed(0)
    f = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    aligned = clip_loss(f, f, torch.tensor(20.0))
    shuffled = clip_loss(f, f[torch.randperm(8)], torch.tensor(20.0))
    assert aligned < shuffled


def test_padding_does_not_change_features():
    """Short and long captions share a batch, so the pad mask has to work."""
    tok, model = build()
    model.eval()
    a, _ = tok.encode("a dog swims in a lake")
    with torch.no_grad():
        f1 = model.encode_text(torch.tensor([a]))
        f2 = model.encode_text(torch.tensor([a, tok.encode(CAPTIONS[0])[0]]))[:1]
    assert torch.allclose(f1, f2, atol=1e-5)


def test_recall_on_perfect_features():
    img = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    txt = img.repeat_interleave(2, dim=0)  # 2 captions per image, identical to it
    text2img = torch.tensor([0, 0, 1, 1, 2, 2])
    m = retrieval_metrics(img, txt, text2img, ks=(1,))
    assert m["t2i_R@1"] == 100.0 and m["i2t_R@1"] == 100.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
