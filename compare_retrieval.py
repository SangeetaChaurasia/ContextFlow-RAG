import json

from hybrid_retriever import search
from reranker import rerank
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

index = pc.Index(
    "pinecone-hybrid-learning"
)

namespace = "knowledge-hybrid-v1"

dense_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

with open(
    "evaluation_dataset.json",
    "r",
    encoding="utf-8"
) as file:
    dataset = json.load(file)

def dense_search(question, top_k=5):
    vector = dense_model.encode(
        question
    ).tolist()

    return index.query(
        namespace=namespace,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

def get_titles(results):
    titles = []

    for match in results["matches"]:
        title = match["metadata"].get(
            "title"
        )

        titles.append(title)

    return titles

def contains_expected(
    titles,
    expected_source,
):
    if expected_source is None:
        return False

    return expected_source in titles

strategies = {
    "Dense": [],
    "Hybrid": [],
    "Hybrid + Reranker": [],
}

for test in dataset:
    question = test["question"]

    expected_source = test.get(
        "expected_source"
    )

    test_type = test.get(
        "type",
        "answerable"
    )

    print("\n")
    print("=" * 70)
    print("QUESTION:", question)
    print("EXPECTED:", expected_source)
    print("TYPE:", test_type)
    print("=" * 70)

    dense_results = dense_search(
        question,
        top_k=5
    )

    dense_titles = get_titles(
        dense_results
    )

    print("\nDense:")
    print(dense_titles)

    if test_type == "answerable":

        strategies["Dense"].append(
            contains_expected(
                dense_titles,
                expected_source
            )
        )

    # -----------------------------------
    # Hybrid
    # -----------------------------------

    hybrid_results = search(
        question,
        top_k=5,
        alpha=0.5,
    )

    hybrid_titles = get_titles(
        hybrid_results
    )

    print("\nHybrid:")
    print(hybrid_titles)

    if test_type == "answerable":

        strategies["Hybrid"].append(
            contains_expected(
                hybrid_titles,
                expected_source
            )
        )


    # -----------------------------------
    # Hybrid + Reranker
    # -----------------------------------

    reranked_results = rerank(
        question,
        hybrid_results,
        top_k=5,
    )

    reranked_titles = [
        result["metadata"].get("title")
        for result in reranked_results
    ]

    print("\nHybrid + Reranker:")
    print(reranked_titles)

    if test_type == "answerable":

        strategies["Hybrid + Reranker"].append(
            contains_expected(
                reranked_titles,
                expected_source
            )
        )


# -----------------------------------
# Final results
# -----------------------------------

print("\n")
print("=" * 70)
print("RETRIEVAL STRATEGY COMPARISON")
print("=" * 70)

total = len(
    strategies["Dense"]
)

for name, results in strategies.items():

    recall_at_5 = (
        sum(results) / total
        if total
        else 0
    )

    print(
        f"{name:22} "
        f"{sum(results)}/{total} "
        f"Recall@5 = "
        f"{recall_at_5:.2%}"
    )
