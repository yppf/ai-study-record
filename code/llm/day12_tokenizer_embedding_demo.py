from collections import Counter

import torch
import torch.nn as nn


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    texts = [
        "large language models learn patterns",
        "models turn tokens into embeddings",
        "embeddings become model inputs",
    ]

    section("1. Tiny word-level tokenizer")
    tokens = [text.split() for text in texts]
    vocab = ["<pad>", "<unk>"] + sorted(Counter(word for sent in tokens for word in sent))
    stoi = {token: idx for idx, token in enumerate(vocab)}
    encoded = [[stoi.get(word, stoi["<unk>"]) for word in sent] for sent in tokens]
    print("vocab =", vocab)
    print("encoded =", encoded)

    section("2. Padding to a batch")
    max_len = max(len(sent) for sent in encoded)
    padded = [sent + [stoi["<pad>"]] * (max_len - len(sent)) for sent in encoded]
    batch = torch.tensor(padded)
    print("batch shape =", tuple(batch.shape))
    print(batch)

    section("3. Embedding lookup")
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4)
    embedded = embedding(batch)
    print("embedded shape =", tuple(embedded.shape))
    print("token id 0 is <pad>")
    print("embedding[0] =", embedding.weight[0])

    section("4. What this means for LLMs")
    print("tokenizer: text -> ids")
    print("embedding: ids -> dense vectors")
    print("the model reads embeddings, not raw strings")


if __name__ == "__main__":
    main()
