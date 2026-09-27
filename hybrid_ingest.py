import os
import re
import json
import hashlib

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from utils.chunker import chunk_text
from sparse_model import encode_sparse

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError(
        "PINECONE_API_KEY"
    )

index_name = "pinecone-hybrid-learning"
namespace = "knowledge-hybrid-v1"

pc = Pinecone(api_key=api_key)

index = pc.Index(index_name)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

document_manifest_file = "data/document_manifest.json"

if os.path.exists(document_manifest_file):
    try:
        with open(document_manifest_file, "r", encoding="utf-8") as f:
            old_hashes = json.load(f)
    except (json.JSONDecodeError, OSError):
        old_hashes = {}
else:
    old_hashes = {}

def get_document_hash(document):
    content = (
        document["title"]
        + document["category"]
        + document["text"]
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

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
            "text": content,
        })

    return documents

with open(
    "data/knowledge.txt",
    "r",
    encoding="utf-8"
) as f:

    content = f.read()

documents = parse_documents(content)

print(
    f"Total documents: {len(documents)}"
)

new_hashes = {}

for document in documents:
    new_hashes[document['title']] = get_document_hash(document)

changed_documents = []
unchanged_documents = []

for document in documents:
    title = document["title"]

    if old_hashes.get(title) == new_hashes[title]:
        unchanged_documents.append(document)
    else:
        changed_documents.append(document)

print(
    f"Changed documents: {len(changed_documents)}"
)

print(
    f"Unchanged documents: {len(unchanged_documents)}"
)

for document in changed_documents:
    print("CHANGED:", document["title"])

for document in unchanged_documents:
    print("UNCHANGED:", document["title"])

chunk_manifest_file = "data/chunk_manifest.json"

if os.path.exists(chunk_manifest_file):
    with open(
        chunk_manifest_file,
        "r",
        encoding="utf-8"
    ) as f:
        old_manifest = json.load(f)
else:
    old_manifest = {}

chunks_data = []
new_manifest = {}
deleted_documents = []
changed_titles = {
    document["title"]
    for document in changed_documents
}

for document in documents:
    title = document["title"]

    # Unchanged document: keep its existing chunk IDs
    if title not in changed_titles:
        new_manifest[title] = old_manifest.get(
            title,
            []
        )
        continue

    # Changed/new document: regenerate chunks
    new_manifest[title] = []

    chunks = chunk_text(
        document["text"],
        chunk_size=500,
        chunk_overlap=100,
    )

    for chunk_index, chunk in enumerate(chunks):
        document_key = re.sub(
            r"[^a-z0-9]+",
            "-",
            title.lower()
        ).strip("-")

        chunk_id = (
            f"{document_key}-chunk-{chunk_index}"
        )

        new_manifest[title].append(chunk_id)

        chunk_data = {
            "id": chunk_id,
            "text": chunk,
            "title": title,
            "category": document["category"],
        }

        if document["category"] == "orders":
            match = re.search(
                r"\bORD-\d+\b",
                title
            )

            if match:
                chunk_data["order_id"] = match.group(0)

        chunks_data.append(chunk_data)

for document_title in old_manifest:
    if document_title not in new_manifest:
        deleted_documents.append(document_title)

print(
    f"Deleted documents: {len(deleted_documents)}"
)

for document_title in deleted_documents:
    print(
        "DELETED:",
        document_title
    )

print(f"Total chunks: {len(chunks_data)}")

records = []

if not chunks_data:
    print("No documents changed. Nothing to embed or upsert.")
else:
    texts = [
        item["text"]
        for item in chunks_data
    ]

    dense_vectors = model.encode(
        texts
    ).tolist()

    print(
        f"Generated dense vectors: {len(dense_vectors)}"
    )

    records = []

    for chunk_data, dense_vector in zip(
        chunks_data,
        dense_vectors
    ):
        text = chunk_data["text"]

        sparse_vector = encode_sparse(text)

        metadata = {
            "text": text,
            "title": chunk_data["title"],
            "category": chunk_data["category"],
            "source": "knowledge.txt",
            "company_id": "company-1",
        }

        if "order_id" in chunk_data:
            metadata["order_id"] = chunk_data["order_id"]

        record = {
            "id": chunk_data["id"],
            "values": dense_vector,
            "sparse_values": sparse_vector,
            "metadata": metadata,
        }

        records.append(record)

    print(f"Total records: {len(records)}")

stale_chunk_ids = []

for document_title, old_chunk_ids in old_manifest.items():

    new_chunk_ids = new_manifest.get(
        document_title,
        []
    )

    for chunk_id in old_chunk_ids:
        if chunk_id not in new_chunk_ids:
            stale_chunk_ids.append(chunk_id)

if stale_chunk_ids:
    print(
        f"Deleting stale chunks: {stale_chunk_ids}"
    )

    index.delete(
        ids=stale_chunk_ids,
        namespace=namespace,
    )
else:
    print("No stale chunks found.")

if records:
    index.upsert(
        vectors=records,
        namespace=namespace,
    )

    print(
        "Changed documents inserted into Pinecone successfully."
    )
else:
    print("No changes to insert into Pinecone.")

print(
    "Hybrid knowledge base inserted into Pinecone successfully."
)

with open(document_manifest_file, "w", encoding="utf-8") as f:
    json.dump(
        new_hashes,
        f,
        indent=4
    )

with open(
    chunk_manifest_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        new_manifest,
        f,
        indent=4
    )
