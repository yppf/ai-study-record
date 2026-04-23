import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    a = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)

    section("1. Dot product")
    print("x =", x)
    print("y =", y)
    print("dot(x, y) =", torch.dot(x, y).item())

    section("2. Matrix-vector product")
    print("a:\n", a)
    print("a shape =", tuple(a.shape))
    print("x shape =", tuple(x.shape))
    print("a @ x =", a @ x)

    section("3. Matrix-matrix product")
    print("b:\n", b)
    print("b shape =", tuple(b.shape))
    print("a @ b shape =", tuple((a @ b).shape))
    print(a @ b)

    section("4. Sum over axes")
    print("sum all =", a.sum().item())
    print("sum dim=0 =", a.sum(dim=0))
    print("sum dim=1 =", a.sum(dim=1))

    section("5. Norms")
    print("L1 norm of x =", torch.abs(x).sum().item())
    print("L2 norm of x =", torch.norm(x).item())
    print("Frobenius norm of a =", torch.norm(a).item())

    section("6. Shape intuition")
    print("If A is (m, n) and B is (n, p), then A @ B is (m, p).")
    print("Here A is", tuple(a.shape), "and B is", tuple(b.shape))
    print("So A @ B is", tuple((a @ b).shape))


if __name__ == "__main__":
    main()
