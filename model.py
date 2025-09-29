import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ---------------------------
# Rotary positional embedding
# ---------------------------

class Rotary(nn.Module):
    """
    Rotary positional embeddings with caching (no external imports beyond torch).
    Produces cos/sin tensors shaped to be applied to qkv of shape (B, T, 3, H, D).
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        """
        x: a tensor whose seq length is at dimension `seq_dim` (we only use that to cache)
        returns (cos, sin) shaped (1, seq_len, 3, 1, dim) which will broadcast across batch/head dims
        """
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())  # (seq_len, dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

            # shape (1, seq_len, 1, 1, dim)
            base_cos = emb.cos()[None, :, None, None, :]
            base_sin = emb.sin()[None, :, None, None, :]

            # expand to 3 for q,k,v axis: (1, seq_len, 3, 1, dim)
            cos = base_cos.repeat(1, 1, 3, 1, 1)
            sin = base_sin.repeat(1, 1, 3, 1, 1)

            # make the value slot an identity transform (no rotation)
            cos[:, :, 2, :, :].fill_(1.0)
            sin[:, :, 2, :, :].fill_(0.0)

            self.cos_cached = cos
            self.sin_cached = sin

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    """rotate_half as in many RoPE implementations"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    """
    qkv: (B, T, 3, H, D)
    cos, sin: (1, T, 3, 1, D)  (or broadcastable)
    returns same shape qkv with rotary applied to q and k (v left unchanged by cos/sin identity)
    """
    return (qkv * cos) + (rotate_half(qkv) * sin)


# ---------------------------
# Transformer blocks
# ---------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # flash attention detection
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size)
            )

        # rotary module (per head dim)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # compute qkv and reshape to (B, T, 3, heads, head_dim)
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim)

        # get cos & sin and apply rotary to q and k (v is left unchanged by design)
        cos, sin = self.rotary(qkv, seq_dim=1)  # shapes broadcastable to qkv
        qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        # split back to q, k, v and move heads forward
        q, k, v = qkv.unbind(dim=2)  # each is (B, T, heads, head_dim)
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # causal self-attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels (PyTorch >= 2.0)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Note: we removed learned positional embeddings (wpe) — rotary is applied in attention
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        Simpler: count everything (we don't have wpe to subtract).
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)   # no learned pos embeddings; rotary applied in attention

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # inference-time: only last position
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load weights from HF GPT-2 where possible.
        This is made tolerant to missing keys (e.g. we've removed wpe).
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # safe-copy HF params into our model if keys exist and shapes match (handle Conv1D->Linear transpose)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k_hf, v_hf in sd_hf.items():
            if k_hf not in sd:
                # skip keys our model doesn't have (e.g. position embeddings wpe)
                continue
            try:
                if any(k_hf.endswith(w) for w in transposed):
                    if sd[k_hf].shape == v_hf.t().shape:
                        sd[k_hf].copy_(v_hf.t())
                    else:
                        print(f"Skipping (transpose) {k_hf}: shape mismatch {sd[k_hf].shape} vs {v_hf.t().shape}")
                else:
                    if sd[k_hf].shape == v_hf.shape:
                        sd[k_hf].copy_(v_hf)
                    else:
                        print(f"Skipping {k_hf}: shape mismatch {sd[k_hf].shape} vs {v_hf.shape}")
            except Exception as e:
                print(f"Failed to copy {k_hf}: {e}")

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
