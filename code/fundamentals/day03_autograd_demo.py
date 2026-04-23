import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. Basic autograd")
    x = torch.arange(4.0, requires_grad=True)
    y = 2 * torch.dot(x, x)
    y.backward()
    print("x =", x)
    print("y =", y.item())
    print("x.grad =", x.grad)
    print("expected =", 4 * x.detach())

    section("2. Clear old gradients")
    x.grad.zero_()
    z = x.sum()
    z.backward()
    print("z =", z.item())
    print("gradient of sum(x) =", x.grad)

    section("3. detach")
    x.grad.zero_()
    u = x.detach()
    v = u * x
    v.sum().backward()
    print("u = x.detach() means u has no gradient history")
    print("gradient after (u * x).sum() =", x.grad)
    print("expected =", u)

    section("4. Tiny training-style example")
    w = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    feature = torch.tensor([3.0])
    label = torch.tensor([10.0])

    pred = w * feature + b
    loss = (pred - label) ** 2
    loss.backward()

    print("pred =", pred.item())
    print("loss =", loss.item())
    print("dw =", w.grad.item())
    print("db =", b.grad.item())

    section("5. What to think about")
    print("Try changing feature, label, w, and b.")
    print("Then predict whether dw and db should be positive or negative.")


if __name__ == "__main__":
    main()
