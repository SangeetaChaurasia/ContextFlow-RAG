import json

from hybrid_retriever import search
from reranker import rerank


# -----------------------------------
# Load evaluation dataset
# -----------------------------------

with open(
    "evaluation_dataset.json",
    "r",
    encoding="utf-8"
) as file:

    dataset = json.load(file)


# -----------------------------------
# Find rank
# -----------------------------------

def find_rank(titles, expected_source):

    if expected_source is None:
        return None

    try:
        return titles.index(expected_source) + 1

    except ValueError:
        return None


# -----------------------------------
# Analyze each question
# -----------------------------------

for test in dataset:

    if test.get("type") != "answerable":
        continue

    question = test["question"]

    expected_source = test.get(
        "expected_source"
    )

    print("\n")
    print("=" * 70)
    print("QUESTION:")
    print(question)

    print("\nEXPECTED SOURCE:")
    print(expected_source)

    print("=" * 70)


    # -----------------------------------
    # Hybrid retrieval
    # -----------------------------------

    results = search(
        question,
        top_k=10,
        alpha=0.5,
    )

    titles = [
        match["metadata"].get("title")
        for match in results["matches"]
    ]

    rank = find_rank(
        titles,
        expected_source
    )


    print("\nHybrid Top 10:")

    for position, title in enumerate(
        titles,
        start=1
    ):
        print(
            f"{position}. {title}"
        )


    print("\nExpected Rank:")

    if rank:
        print(rank)

    else:
        print("NOT FOUND")


    # -----------------------------------
    # Reranking
    # -----------------------------------

    reranked = rerank(
        question,
        results,
        top_k=10,
    )

    reranked_titles = [
        result["metadata"].get("title")
        for result in reranked
    ]

    rerank_rank = find_rank(
        reranked_titles,
        expected_source
    )


    print("\nReranked Top 10:")

    for position, title in enumerate(
        reranked_titles,
        start=1
    ):
        print(
            f"{position}. {title}"
        )


    print("\nReranked Expected Rank:")

    if rerank_rank:
        print(rerank_rank)

    else:
        print("NOT FOUND")


    # -----------------------------------
    # Diagnosis
    # -----------------------------------

    print("\nDiagnosis:")

    if rank is None:

        print(
            "RETRIEVAL FAILURE: "
            "Expected document was not retrieved."
        )

    elif rank > 5:

        print(
            "RETRIEVAL WEAK: "
            f"Expected document is at rank {rank}."
        )

    else:

        print(
            f"Retrieval OK: expected document "
            f"is at rank {rank}."
        )


    if rank and rerank_rank:

        if rerank_rank < rank:

            print(
                f"Reranker IMPROVED rank "
                f"{rank} -> {rerank_rank}"
            )

        elif rerank_rank > rank:

            print(
                f"Reranker WORSENED rank "
                f"{rank} -> {rerank_rank}"
            )

        else:

            print(
                "Reranker did not change the rank."
            )
