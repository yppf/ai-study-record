import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. Scalars, vectors, matrices, tensors")
    scalar = torch.tensor(3.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    print("scalar:", scalar, "shape=", tuple(scalar.shape))
    print("vector:", vector, "shape=", tuple(vector.shape))
    print("matrix:\n", matrix, "shape=", tuple(matrix.shape))
    print("tensor shape=", tuple(tensor.shape), "numel=", tensor.numel())

    section("2. Reshape and view")
    x = torch.arange(12, dtype=torch.float32)
    print("original:", x, "shape=", tuple(x.shape))
    x2 = x.reshape(3, 4)
    print("reshaped:\n", x2)

    section("3. Broadcasting")
    a = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    b = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    c = a + b
    print("a shape=", tuple(a.shape))
    print("b shape=", tuple(b.shape))
    print("a + b shape=", tuple(c.shape))
    print(c)

    section("4. Indexing and slicing")
    print("x2[1, 2] =", x2[1, 2].item())
    print("x2[0:2, 1:3]:\n", x2[0:2, 1:3])

    section("5. Reduction")
    print("sum all =", x2.sum().item())
    print("sum axis 0 =", x2.sum(dim=0))
    print("sum axis 1 =", x2.sum(dim=1))

    section("6. What to think about")
    print("Try changing:")
    print("- reshape(2, 6) or reshape(4, 3)")
    print("- a shape to (3, 1) and b shape to (1, 5)")
    print("- slicing ranges and predict the output shape first")


if __name__ == "__main__":
    main()
