import math
import torch
import torch.nn as nn
import torch.nn.functional as F

GPT2_CONFIG = {
    "vocab_size": 384,
    "seq_len": 128,
    "emb_dim": 512,
    "n_head": 8,      # 512 % 8 == 0，必须整除
    "n_layer": 6,
    "dropout": 0.1,
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        n_head = cfg["n_head"]
        seq_len = cfg["seq_len"]

        assert emb_dim % n_head == 0, "emb_dim must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = emb_dim // n_head

        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim)   ## 一次性计算 q, k, v
        self.c_proj = nn.Linear(emb_dim, emb_dim)       ## 输出线性变换

        self.attn_drop = nn.Dropout(cfg["dropout"])
        self.resid_drop = nn.Dropout(cfg["dropout"])

        # 因果 mask：只看当前位置及之前
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len), persistent=False)

    def forward(self, x):
        b, t, c = x.size()

        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(c, dim=2)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))      # 应用 mask，禁止未来信息
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # (B, T, C)

        y = self.c_proj(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.fc = nn.Linear(emb_dim, 4 * emb_dim)
        self.proj = nn.Linear(4 * emb_dim, emb_dim)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.ln_1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["seq_len"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["dropout"])

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["emb_dim"])

        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 权重共享（GPT 常见做法）
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token ids
        targets: (B, T) token ids, 可选，用于计算训练 loss
        """
        b, t = idx.size()
        if t > self.cfg["seq_len"]:
            raise ValueError(f"Sequence length {t} exceeds model max {self.cfg['seq_len']}")

        pos = torch.arange(0, t, device=idx.device)  # (T,)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]  # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["seq_len"]:]  # 只保留最后 seq_len
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    model = GPT(GPT2_CONFIG)
    x = torch.randint(0, GPT2_CONFIG["vocab_size"], (2, 32))
    y = torch.randint(0, GPT2_CONFIG["vocab_size"], (2, 32))
    logits, loss = model(x, y)
    print("logits:", logits.shape)  # (2, 32, vocab_size)
    print("loss:", float(loss))