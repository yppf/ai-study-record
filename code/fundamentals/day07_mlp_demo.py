import torch
import torch.nn.functional as F


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    torch.manual_seed(0)
    x = torch.tensor([[1.0, -1.0, 2.0]])
    w1 = torch.randn(3, 4)
    b1 = torch.randn(4)
    w2 = torch.randn(4, 2)
    b2 = torch.randn(2)

    section("1. Without activation, two layers collapse into one linear map")
    hidden_linear = x @ w1 + b1
    out_linear = hidden_linear @ w2 + b2
    combined_w = w1 @ w2
    combined_b = b1 @ w2 + b2
    out_single = x @ combined_w + combined_b
    print("two linear layers output =", out_linear)
    print("single combined layer output =", out_single)
    print("difference =", (out_linear - out_single).abs().max().item())

    section("2. Add nonlinearity")
    hidden_relu = F.relu(x @ w1 + b1)
    out_relu = hidden_relu @ w2 + b2
    print("relu hidden =", hidden_relu)
    print("relu network output =", out_relu)

    section("3. Compare activations")
    preact = x @ w1 + b1
    print("pre-activation =", preact)
    print("relu =", F.relu(preact))
    print("sigmoid =", torch.sigmoid(preact))
    print("tanh =", torch.tanh(preact))

    section("4. What to try next")
    print("Try deleting ReLU and verify the network becomes linear again.")
    print("Try changing one negative pre-activation into a positive one.")


if __name__ == "__main__":
    main()
