import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured")

pc = Pinecone(api_key=api_key)

index_name = "pinecone-hybrid-learning"

existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    print(f"Created index: {index_name}")

else:
    print(f"Index already exists: {index_name}")

index = pc.Index(index_name)

print(index.describe_index_stats())
