import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sparse_model import encode_sparse

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured.")

pc = Pinecone(api_key=api_key)

index = pc.Index(
    "pinecone-hybrid-learning"
)

namespace = "knowledge-hybrid-v1"

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

question = "What is the status of order ORD-83921?"

print(f"\nQuestion: {question}")

dense_vector = model.encode(
    question
).tolist()

sparse_vector = encode_sparse(
    question
)

print("\nSparse vector:")
print(sparse_vector)

def scale_vector(vector, alpha):
    return [
        value * alpha
        for value in vector
    ]

alpha = 0.5

results = index.query(
    namespace=namespace,
    vector=scale_vector(
        dense_vector,
        alpha
    ),
    sparse_vector={
        "indices": sparse_vector["indices"],
        "values": [
            value * (1 - alpha)
            for value in sparse_vector["values"]
        ],
    },
    top_k=3,
    include_metadata=True,
)

print("\n" + "=" * 60)
print("HYBRID SEARCH")
print("=" * 60)

for match in results["matches"]:
    print(
        f"\nID: {match['id']}"
    )

    print(
        f"Score: {match['score']}"
    )

    metadata = match.get(
        "metadata",
        {}
    )

    print(
        f"Title: {metadata.get('title')}"
    )

    print(
        f"Title: {metadata.get('category')}"
    )

    print(
        f"Text: {metadata.get('text')}"
    )

    print("-" * 60)
