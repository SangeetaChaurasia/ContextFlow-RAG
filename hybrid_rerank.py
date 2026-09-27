from hybrid_retriever import search
from reranker import rerank


question = "Is ORD-83921 delivered?"


# -----------------------------------
# Step 1: Hybrid retrieval
# -----------------------------------

results = search(
    question,
    top_k=5,
    alpha=0.5,
)


print("\n")
print("=" * 60)
print("PINECONE HYBRID RESULTS")
print("=" * 60)


for match in results["matches"]:

    metadata = match["metadata"]

    print(
        f"\nID: {match['id']}"
    )

    print(
        f"Pinecone Score: {match['score']}"
    )

    print(
        f"Title: {metadata.get('title')}"
    )

    print(
        f"Text: {metadata.get('text')}"
    )

    print("-" * 60)


# -----------------------------------
# Step 2: Reranking
# -----------------------------------

reranked_results = rerank(
    question,
    results,
    top_k=3,
)


print("\n")
print("=" * 60)
print("RERANKED RESULTS")
print("=" * 60)


for result in reranked_results:

    metadata = result["metadata"]

    print(
        f"\nID: {result['id']}"
    )

    print(
        f"Reranker Score: {result['score']}"
    )

    print(
        f"Title: {metadata.get('title')}"
    )

    print(
        f"Category: {metadata.get('category')}"
    )

    print(
        f"Text: {metadata.get('text')}"
    )

    print("-" * 60)
