import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    torch.manual_seed(42)

    section("1. Synthetic data")
    num_examples = 200
    num_features = 2
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    features = torch.randn(num_examples, num_features)
    labels = features @ true_w + true_b
    labels += 0.01 * torch.randn_like(labels)
    print("features shape =", tuple(features.shape))
    print("labels shape =", tuple(labels.shape))

    section("2. Training loop")
    w = torch.randn(num_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.05

    def linreg(x: torch.Tensor) -> torch.Tensor:
        return x @ w + b

    for epoch in range(100):
        pred = linreg(features)
        loss = ((pred - labels) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        if epoch in {0, 4, 9, 19, 49, 99}:
            print(f"epoch={epoch + 1}, loss={loss.item():.6f}")

    section("3. Learned parameters")
    print("true_w =", true_w)
    print("learned_w =", w.detach())
    print("true_b =", true_b)
    print("learned_b =", b.item())

    section("4. What to try next")
    print("Try changing lr to 0.3 or 0.003 and compare convergence.")
    print("Try changing num_features or noise level.")


if __name__ == "__main__":
    main()
