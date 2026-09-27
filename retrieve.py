import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured")

pc = Pinecone(api_key=api_key)

index = pc.Index("pinecone-semantic-learning")

question = "What is the status of order ORD-74125?"

results = index.search(
    namespace="knowledge-v5",
    inputs={
        "text": question
    },
    top_k=10,
    rerank={
        "model": "bge-reranker-v2-m3",
        "rank_fields": ["text"],
        "top_n": 3,
    }
)

print(f"Results for question: {question}")

print("\nReranked results:")

for hit in results.result.hits:
    print("ID:", hit.id)
    print("Score:", hit.score)
    print("Text:", hit.fields.get("text"))
    print("Title:", hit.fields.get("title"))
    print("Category:", hit.fields.get("category"))
    print("-" * 60)
