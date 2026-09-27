import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured")

pc = Pinecone(api_key=api_key)

index = pc.Index("pinecone-semantic-learning")

results = index.search(
    namespace="articles",
    inputs={
        "text": "I need to change my email"
    },
    top_k=3,
    filter={
        "category": {
            "$in": ["account", "billing"]
        }
    }
)

for hit in results.result.hits:
    print("ID:", hit.id)
    print("Score:", hit.score)
    print("Text:", hit.fields.get("text"))
    print("Category:", hit.fields.get("category"))
    print("-" * 50)