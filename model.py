import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    #kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, max_seq_len: int, device: str, rope_theta: float = 10000.0):
    
    assert head_dim % 2 == 0, "head_dim must be even for rope"
    
    # if dim = 256
    theta_positions = torch.arange(0, head_dim, 2).float()
    # theta_i = B**(-2i/dim)
    theta_values = 1.0 / (rope_theta**(theta_positions/head_dim)).to(device).view(1, head_dim//2) # head_dim / 2 thetas
    m = torch.arange(max_seq_len, device=device).float().view(max_seq_len, 1) # max_seqlen, 1
    token_thetas = m @ theta_values # max_seqlen, head_dim / 2

    # basically, creates cis(theta) for every theta, this is, e^(i*theta)
    # torch polar receives (magnitudes, angles), so its creating a bunch of 1*cis(thetas)
    freqs_cis = torch.polar(torch.ones_like(token_thetas), token_thetas)
    return freqs_cis

def apply_rotary_embeddings(x: torch.Tensor, freqs_cis: torch.Tensor, device: str):
    # x is (B , seqlen, n_heads, head_dim)
    # basically turns all into x + yi, after grouping every 2 dimensions
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # becomes B, seqlen, n_heads, head_dim // 2
    
    # (max_seq_len, head_dim/2) - > (1, max_seqlen, 1, head_dim / 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # rotates by the freqs_cis, multiplying by the cis(theta)s
    rotated_complex_x = complex_x * freqs_cis

    rotated_real_x = torch.view_as_real(rotated_complex_x).reshape(*x.shape)
    return rotated_real_x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        # x is (B, seqlen, dim),
        # rms is (B, seqlen, 1)
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1) + self.eps).unsqueeze(-1)

        x_norm = x * rms_inv
        x_weighted = x_norm * self.weight

        return x_weighted

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads_q = args.n_heads
        self.n_heads_kv = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        self.head_dim = args.dim // self.n_heads_q
        self.n_rep = self.n_heads_q // self.n_heads_kv

        self.wq = nn.Linear(args.dim, self.head_dim * self.n_heads_q, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_heads_kv, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_heads_kv, bias=False)
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor):
        # x being shaped (B, 1, dim)
        B, T, _ = x.shape

        # (b, 1, n_heads * head_dim)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(B, T, self.n_heads_q, self.head_dim)
        xk = xk.view(B, T, self.n_heads_kv, self.head_dim)
        xv = xv.view(B, T, self.n_heads_kv, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_cis, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_cis, device=x.device)


        self.cache_k[:B, start_pos:start_pos+T] = xk
        self.cache_v[:B, start_pos:start_pos+T] = xv
        
        xk = self.cache_k[:B, :start_pos + T]
        xv = self.cache_v[:B, :start_pos + T]
        
        # (b, t, n_heads_kv, head_dim) so far, goal: (b, t, n_heads_q, head_dim), how? Replicating n_rep times
        xq = xq.permute(0, 2, 1, 3) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        xk = xk.repeat_interleave(self.n_rep, dim=2).permute(0, 2, 3, 1) # (b, n_heads_q, head_dim, t)
        xv = xv.repeat_interleave(self.n_rep, dim=2).permute(0, 2, 1, 3) # (b, n_heads_q, t, head_dim)

        scores = torch.matmul(xq, xk) / math.sqrt(self.head_dim)

        # So, scores are (B, n_heads, 1, T), because there will be only 1 query,
        # hence, we don't need tril mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # as values are (B, n_heads, T, head_dim)
        attended = torch.matmul(scores, xv)
        # attended is (B, n_heads, 1, head_dim)
        pre_out = attended.squeeze(2).view(B, self.n_heads_q * self.head_dim)

        return self.wo(pre_out)
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        # B, T, dim -> B, T, x*dim
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        
        x = self.w2(swish * x_V)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)

        self.feed_forward = FeedForward(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x):
        attended_x = self.attention(self.attention_norm(x))
        out = self.ffw_norm = self.feed_forward(self.ffn_norm(x))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set, got -1"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        #tokens -> B, 1 because using kv cache
        batch_size, seqlen = tokens.shape
        assert seqlen == 1, "Expecting one token at a time for kv_cache"

        embeddings = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos: start_pos + seqlen]

        layer_x = embeddings
        for layer in self.layers:
            layer_x = layer(layer_x, start_pos, freqs_complex)
        x_normalized = self.norm(layer_x)
        output = self.output(x_normalized).float()
        return output
