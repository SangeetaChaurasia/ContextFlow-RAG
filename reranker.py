from sentence_transformers import CrossEncoder


model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(question, results, top_k=3):

    pairs = []

    for match in results["matches"]:

        text = match["metadata"].get(
            "text",
            ""
        )

        pairs.append([
            question,
            text
        ])


    scores = model.predict(pairs)


    reranked = []

    for match, score in zip(
        results["matches"],
        scores
    ):

        reranked.append({
            "id": match["id"],
            "score": float(score),
            "metadata": match["metadata"],
        })


    reranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    return reranked[:top_k]
