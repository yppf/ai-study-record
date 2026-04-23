import torch
import torch.nn as nn


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def train_model(weight_decay: float) -> torch.Tensor:
    torch.manual_seed(7)
    x = torch.randn(128, 20)
    true_w = torch.zeros(20)
    true_w[:3] = torch.tensor([2.0, -1.5, 0.7])
    y = x @ true_w + 0.1 * torch.randn(128)

    w = torch.randn(20, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.05

    for _ in range(80):
        pred = x @ w + b
        loss = ((pred - y) ** 2).mean() + 0.5 * weight_decay * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.detach()


def main() -> None:
    section("1. Weight decay effect")
    w_no_decay = train_model(weight_decay=0.0)
    w_decay = train_model(weight_decay=0.1)
    print("weight norm without decay =", torch.norm(w_no_decay).item())
    print("weight norm with decay =", torch.norm(w_decay).item())

    section("2. Dropout train vs eval")
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(10)
    dropout.train()
    print("dropout(train) =", dropout(x))
    dropout.eval()
    print("dropout(eval) =", dropout(x))

    section("3. Backprop through a tiny MLP")
    net = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    feature = torch.tensor([[1.0, 2.0, -1.0]])
    label = torch.tensor([[2.5]])
    pred = net(feature)
    loss = ((pred - label) ** 2).mean()
    loss.backward()
    print("loss =", loss.item())
    print("grad norm layer1 =", net[0].weight.grad.norm().item())
    print("grad norm layer2 =", net[2].weight.grad.norm().item())

    section("4. What to try next")
    print("Try changing weight_decay to 1.0 and compare the weight norm.")
    print("Run dropout(train) several times and compare randomness.")


if __name__ == "__main__":
    main()
