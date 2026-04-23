import math

import torch
import torch.nn as nn


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def main() -> None:
    torch.manual_seed(0)
    batch_size, seq_len, d_model = 2, 4, 8
    num_heads = 2

    section("1. Input and positional encoding")
    x = torch.randn(batch_size, seq_len, d_model)
    pe = positional_encoding(seq_len, d_model)
    x = x + pe.unsqueeze(0)
    print("input shape =", tuple(x.shape))
    print("positional encoding shape =", tuple(pe.shape))

    section("2. Causal mask")
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print("causal mask shape =", tuple(causal_mask.shape))
    print(causal_mask)

    section("3. Multi-head self-attention")
    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
    attn_output, attn_weights = mha(
        x, x, x, attn_mask=causal_mask, need_weights=True, average_attn_weights=False
    )
    print("attn output shape =", tuple(attn_output.shape))
    print("attn weights shape =", tuple(attn_weights.shape))

    section("4. AddNorm + FFN")
    norm1 = nn.LayerNorm(d_model)
    norm2 = nn.LayerNorm(d_model)
    ffn = nn.Sequential(
        nn.Linear(d_model, d_model * 2),
        nn.ReLU(),
        nn.Linear(d_model * 2, d_model),
    )
    y = norm1(x + attn_output)
    z = norm2(y + ffn(y))
    print("after first AddNorm =", tuple(y.shape))
    print("after FFN and second AddNorm =", tuple(z.shape))

    section("5. What to try next")
    print("Try changing num_heads from 2 to 4.")
    print("Try removing the causal mask and compare attention behavior.")


if __name__ == "__main__":
    main()
