"""Word-level tokenizer. Nothing fancy, just enough for Flickr8k captions."""

import json
import re
from collections import Counter

# keeps words and apostrophes, drops everything else
_TOKEN_RE = re.compile(r"[a-z0-9']+")

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class SimpleTokenizer:
    def __init__(self, itos, context_length=32):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.context_length = context_length
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def build(cls, captions, min_freq=2, max_size=None, context_length=32):
        counts = Counter()
        for c in captions:
            counts.update(tokenize(c))

        words = [w for w, n in counts.most_common() if n >= min_freq]
        if max_size is not None:
            words = words[: max_size - len(SPECIALS)]
        return cls(SPECIALS + words, context_length=context_length)

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        """Returns (ids, length). Always ends with <eos>, padded to context_length."""
        ids = [self.stoi.get(w, self.unk_id) for w in tokenize(text)]
        ids = [self.bos_id] + ids[: self.context_length - 2] + [self.eos_id]
        length = len(ids)
        ids += [self.pad_id] * (self.context_length - length)
        return ids, length

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == EOS:
                break
            if tok in (PAD, BOS):
                continue
            out.append(tok)
        return " ".join(out)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"itos": self.itos, "context_length": self.context_length}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["itos"], context_length=d["context_length"])
