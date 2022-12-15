import torch
import torch.nn as nn
import torch.nn.functional as F

from model_mingpt import GPT


class VQGANTransformer(nn.Module):
    def __init__(self, sos_token=0, pkeep=0.5):
        super(VQGANTransformer, self).__init__()

        self.sos_token = sos_token
        self.pkeep = pkeep

        transformer_config = {
            "vocab_size": 2049,
            "block_size": 256,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 512
        }
        self.transformer = GPT(**transformer_config)

    def forward(self, indices):
        sos_tokens = torch.ones(indices.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x
