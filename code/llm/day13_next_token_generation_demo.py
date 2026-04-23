import torch


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> tuple[int, torch.Tensor]:
    scaled = logits / temperature
    if top_k is not None:
        values, indices = torch.topk(scaled, top_k)
        probs = torch.softmax(values, dim=-1)
        chosen_local = torch.multinomial(probs, num_samples=1).item()
        return indices[chosen_local].item(), probs
    probs = torch.softmax(scaled, dim=-1)
    chosen = torch.multinomial(probs, num_samples=1).item()
    return chosen, probs


def main() -> None:
    vocab = ["你好", "世界", "模型", "学习", "<eos>"]
    logits = torch.tensor([2.5, 1.0, 0.5, 3.0, -1.0])

    section("1. Logits to next-token probabilities")
    probs = torch.softmax(logits, dim=-1)
    print("vocab =", vocab)
    print("logits =", logits)
    print("probs =", probs)
    print("argmax token =", vocab[probs.argmax().item()])

    section("2. Temperature")
    for temperature in [0.5, 1.0, 1.5]:
        temp_probs = torch.softmax(logits / temperature, dim=-1)
        print(f"temperature={temperature}, probs={temp_probs.tolist()}")

    section("3. Top-k sampling")
    torch.manual_seed(0)
    token_id, topk_probs = sample_next_token(logits, temperature=1.0, top_k=2)
    print("top-k probs =", topk_probs)
    print("sampled token =", vocab[token_id])

    section("4. What this means for GPT-style models")
    print("model output: logits over the whole vocabulary")
    print("softmax: logits -> probabilities")
    print("decoding: choose or sample the next token")


if __name__ == "__main__":
    main()
