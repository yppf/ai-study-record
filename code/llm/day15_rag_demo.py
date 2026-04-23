import torch
import torch.nn.functional as F


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def embed(text: str) -> torch.Tensor:
    vec = torch.tensor(
        [
            1.0 if "Transformer" in text or "注意力" in text else 0.0,
            1.0 if "LoRA" in text or "低秩" in text else 0.0,
            1.0 if "RAG" in text or "检索" in text else 0.0,
            1.0 if "模型" in text or "知识" in text else 0.0,
        ],
        dtype=torch.float32,
    )
    vec = vec + 1e-6
    return F.normalize(vec, dim=0)


def main() -> None:
    docs = [
        "Transformer 使用自注意力机制处理序列。",
        "LoRA 通过低秩更新减少微调参数量。",
        "RAG 通过检索外部知识提升回答质量。",
    ]
    query = "什么是 RAG，为什么它能补充模型知识？"

    section("1. Retrieval")
    doc_embeds = torch.stack([embed(doc) for doc in docs])
    query_embed = embed(query)
    sims = doc_embeds @ query_embed
    top_idx = torch.argmax(sims).item()
    print("query =", query)
    print("similarities =", sims.tolist())
    print("top document =", docs[top_idx])

    section("2. Augmented prompt")
    prompt = f"问题: {query}\n参考资料: {docs[top_idx]}\n请基于参考资料作答。"
    print(prompt)

    section("3. Intuition")
    print("RAG = retrieve first, then generate")
    print("the model does not rely only on parametric memory")
    print("retrieval helps ground the answer in external knowledge")


if __name__ == "__main__":
    main()
