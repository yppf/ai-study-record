import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    torch.manual_seed(0)
    in_dim, out_dim, rank = 64, 64, 4
    x = torch.randn(3, in_dim)
    base_w = torch.randn(in_dim, out_dim)

    section("1. Base linear layer")
    base_out = x @ base_w
    print("x shape =", tuple(x.shape))
    print("base_w shape =", tuple(base_w.shape))
    print("base output shape =", tuple(base_out.shape))

    section("2. LoRA low-rank update")
    a = torch.randn(in_dim, rank)
    b = torch.randn(rank, out_dim)
    delta_w = a @ b
    lora_out = x @ (base_w + delta_w)
    print("A shape =", tuple(a.shape))
    print("B shape =", tuple(b.shape))
    print("delta_w shape =", tuple(delta_w.shape))
    print("LoRA output shape =", tuple(lora_out.shape))

    section("3. Parameter comparison")
    full_params = in_dim * out_dim
    lora_params = in_dim * rank + rank * out_dim
    print("full fine-tuning params =", full_params)
    print("LoRA trainable params =", lora_params)
    print("parameter reduction ratio =", round(lora_params / full_params, 4))

    section("4. Intuition")
    print("full fine-tuning updates all weights")
    print("LoRA keeps base weights and learns a low-rank update")
    print("this is why LoRA is cheaper in memory and training cost")


if __name__ == "__main__":
    main()
