import os
import re

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from hybrid_encoder import (build_vocabulary, sparse_encode,)

load_dotenv()


api_key = os.getenv("PINECONE_API_KEY")

index_name = "pinecone-hybrid-learning"
namespace = "knowledge-hybrid-v1"

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

model = SentenceTransformer (
    "all-MiniLM-L6-v2"
)

with open(
    "data/knowledge.txt",
    "r",
    encoding="utf-8"
) as f:

    content = f.read()

def parse_documents(text):
    pattern = r"(?=\[[^\]]+\]\s*\nCategory:)"

    raw_documents = re.split(
        pattern,
        text.strip()
    )

    documents = []

    for document in raw_documents:
        document = document.strip()

        if not document:
            continue

        lines = document.splitlines()
        title = lines[0].strip("[]")

        category = (
            lines[1]
            .replace("Category:", "")
            .strip()
        )

        content = "\n".join(
            lines[3:]
        ).strip()

        documents.append({
            "title": title,
            "category": category,
            "text": content
        })

    return documents

documents = parse_documents(content)

vocabulary = build_vocabulary([
    document["text"]
    for document in documents
])

question = "How can I track my shipment?"


print(
    f"\nQuestion: {question}\n"
)

dense_vector = model.encode(
    question
).tolist()

sparse_vector = sparse_encode(
    question,
    vocabulary,
)

print("Sparse query:")
print(sparse_vector)

dense_results = index.query(
    namespace=namespace,
    vector=dense_vector,
    top_k=3,
    include_metadata=True,
)

print("\n")
print("=" * 60)
print("DENSE SEARCH")
print("=" * 60)

for match in dense_results["matches"]:
    print(f"\nID: {match['id']}")

    print(
        f"Score: {match['score']}"
    )

    print(
        f"Title: "
        f"{match['metadata']['title']}"
    )

    print(
        f"Text: "
        f"{match['metadata']['text']}"
    )

alpha = 0.0

hybrid_dense = [
    value * alpha
    for value in dense_vector
]

hybrid_sparse = {
    "indices": sparse_vector["indices"],
    "values": [
        value * (1- alpha)
        for value in sparse_vector["values"]
    ],
}

hybrid_results = index.query(
    namespace=namespace,
    vector=hybrid_dense,
    sparse_vector=hybrid_sparse,
    top_k=3,
    include_metadata=True,
)

print("\n")
print("=" * 60)
print("HYBRID SEARCH")
print("=" * 60)


for match in hybrid_results["matches"]:

    print(
        f"\nID: {match['id']}"
    )

    print(
        f"Score: {match['score']}"
    )

    print(
        f"Title: "
        f"{match['metadata']['title']}"
    )

    print(
        f"Text: "
        f"{match['metadata']['text']}"
    )
