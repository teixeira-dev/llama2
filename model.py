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
    # x is (B , seqlen, head_dim)
    # basically turns all into x + yi, after grouping every 2 dimensions
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # becomes B, seqlen, head_dim // 2
    
    # (max_seq_len, head_dim/2) - > (1, max_seqlen, 1, head_dim / 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # rotates by the freqs_cis, multiplying by the cis(theta)s
    rotated_complex_x = complex_x * freqs_cis

    rotated_real_x = torch.view_as_real(rotated_complex_x).reshape(*x.shape)
    return rotated_real_x

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
