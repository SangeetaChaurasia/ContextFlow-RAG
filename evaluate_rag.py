import json

from hybrid_retriever import search
from reranker import rerank


def load_dataset():

    with open(
        "evaluation_dataset.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def evaluate_retrieval(
    question,
    expected_source
):

    results = search(
        question=question,
        company_id="company-1",
        top_k=top_k,
        alpha=alpha,
    )

    reranked_results = rerank(
        question,
        results,
        top_k=5,
    )

    sources = []

    for result in reranked_results:

        metadata = result["metadata"]

        title = metadata.get(
            "title",
            "Unknown"
        )

        sources.append(title)

    recall_at_1 = (
        expected_source in sources[:1]
    )

    recall_at_3 = (
        expected_source in sources[:3]
    )

    recall_at_5 = (
        expected_source in sources[:5]
    )

    return (
        recall_at_1,
        recall_at_3,
        recall_at_5,
        sources
    )


def main():

    dataset = load_dataset()

    total = len(dataset)

    recall_1_count = 0
    recall_3_count = 0
    recall_5_count = 0

    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for index, test in enumerate(
        dataset,
        start=1
    ):

        question = test["question"]

        expected_source = test[
            "expected_source"
        ]

        (
            recall_at_1,
            recall_at_3,
            recall_at_5,
            sources
        ) = evaluate_retrieval(
            question,
            expected_source
        )

        if recall_at_1:
            recall_1_count += 1

        if recall_at_3:
            recall_3_count += 1

        if recall_at_5:
            recall_5_count += 1

        print(
            f"\nTest {index}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: {expected_source}"
        )

        print(
            f"Top 5: {sources}"
        )

        print(
            f"Recall@1: "
            f"{'PASS' if recall_at_1 else 'FAIL'}"
        )

        print(
            f"Recall@3: "
            f"{'PASS' if recall_at_3 else 'FAIL'}"
        )

        print(
            f"Recall@5: "
            f"{'PASS' if recall_at_5 else 'FAIL'}"
        )

    recall_at_1 = (
        recall_1_count / total
    ) * 100

    recall_at_3 = (
        recall_3_count / total
    ) * 100

    recall_at_5 = (
        recall_5_count / total
    ) * 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Recall@1: "
        f"{recall_1_count}/{total} "
        f"({recall_at_1:.2f}%)"
    )

    print(
        f"Recall@3: "
        f"{recall_3_count}/{total} "
        f"({recall_at_3:.2f}%)"
    )

    print(
        f"Recall@5: "
        f"{recall_5_count}/{total} "
        f"({recall_at_5:.2f}%)"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
