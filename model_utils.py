import torch
from torch import nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers

def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x
