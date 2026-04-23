import math

import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def optimize(lr: float, momentum: float = 0.0, steps: int = 20) -> tuple[list[float], float]:
    x = torch.tensor([8.0])
    velocity = torch.tensor([0.0])
    history = []

    for _ in range(steps):
        grad = 2 * x
        if momentum > 0:
            velocity = momentum * velocity + grad
            x = x - lr * velocity
        else:
            x = x - lr * grad
        history.append(x.item())
    return history, x.item()


def main() -> None:
    section("1. Objective")
    print("We minimize f(x) = x^2, whose minimum is at x = 0.")

    section("2. SGD with different learning rates")
    for lr in [0.05, 0.2, 0.6]:
        history, final_x = optimize(lr=lr, momentum=0.0)
        print(f"lr={lr}, final_x={final_x:.6f}, first_5={[round(v, 4) for v in history[:5]]}")

    section("3. Momentum")
    history_sgd, final_sgd = optimize(lr=0.1, momentum=0.0)
    history_mom, final_mom = optimize(lr=0.05, momentum=0.8)
    print("plain sgd first_8 =", [round(v, 4) for v in history_sgd[:8]])
    print("momentum first_8 =", [round(v, 4) for v in history_mom[:8]])
    print("final plain sgd =", round(final_sgd, 6))
    print("final momentum =", round(final_mom, 6))

    section("4. Learning rate intuition")
    print("small lr: converges slowly")
    print("good lr: converges steadily")
    print("too large lr: may oscillate or diverge")
    print("momentum: can speed up progress on consistent directions")


if __name__ == "__main__":
    main()
