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


records=[
    {
        "id": "article-1",
        "text": "How to reset your password",
        "category": "account"
    },
    {
        "id": "article-2",
        "text": "How to request a refund",
        "category": "billing"
    },
    {
        "id": "article-3",
        "text": "How to track an order",
        "category": "shipping"
    },
    {
        "id": "article-4",
        "text": "How to update your email address",
        "category": "account",
    },
]

index.upsert_records(
    namespace="articles",
    records=records
)

print("Records submitted successfully")
