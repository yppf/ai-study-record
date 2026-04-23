import torch
import torch.nn.functional as F


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def softmax(logits: torch.Tensor) -> torch.Tensor:
    exps = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return exps / exps.sum(dim=1, keepdim=True)


def main() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 3.0]])
    labels = torch.tensor([0, 2])

    section("1. Logits to probabilities")
    probs = softmax(logits)
    print("logits:\n", logits)
    print("probs:\n", probs)
    print("row sums =", probs.sum(dim=1))

    section("2. Predictions")
    pred = probs.argmax(dim=1)
    print("predicted classes =", pred)
    print("labels =", labels)

    section("3. Cross-entropy")
    picked = probs[torch.arange(len(labels)), labels]
    manual_loss = -torch.log(picked).mean()
    builtin_loss = F.cross_entropy(logits, labels)
    print("p(correct class) =", picked)
    print("manual cross entropy =", manual_loss.item())
    print("torch cross entropy =", builtin_loss.item())

    section("4. What to try next")
    print("Try increasing the correct-class logit and watch loss go down.")
    print("Try changing labels and compare the loss immediately.")


if __name__ == "__main__":
    main()
