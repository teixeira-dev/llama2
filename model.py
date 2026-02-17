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
