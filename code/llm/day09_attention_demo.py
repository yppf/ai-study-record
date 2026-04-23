import math

import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


def main() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    v = torch.tensor([[[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]])

    section("1. Shapes")
    print("q shape =", tuple(q.shape))
    print("k shape =", tuple(k.shape))
    print("v shape =", tuple(v.shape))

    section("2. Attention without mask")
    output, weights = scaled_dot_product_attention(q, k, v)
    print("weights shape =", tuple(weights.shape))
    print("weights =\n", weights)
    print("row sums =", weights.sum(dim=-1))
    print("output =\n", output)

    section("3. Attention with mask")
    mask = torch.tensor([[[1, 1, 0], [1, 0, 0]]])
    masked_output, masked_weights = scaled_dot_product_attention(q, k, v, mask)
    print("mask =\n", mask)
    print("masked weights =\n", masked_weights)
    print("masked output =\n", masked_output)

    section("4. What to try next")
    print("Try changing q and see which key gets more weight.")
    print("Try masking different positions and compare the output.")


if __name__ == "__main__":
    main()
