import os
import re

from dotenv import load_dotenv
from pinecone import Pinecone
from utils.chunker import chunk_text

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured")

pc = Pinecone(api_key=api_key)

index = pc.Index("pinecone-semantic-learning")

def parse_documents(text):
    pattern = r"(?=\[[^\]]+\]\s*\nCategory:)"

    raw_documents = re.split(pattern, text.strip())

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

        content = "\n".join(lines[3:]).strip()

        documents.append({
            "title": title,
            "category": category,
            "text": content,
        })

    return documents

#-----------------------------
# Read Knowledgebase
#-----------------------------
with open("data/knowledge.txt", "r", encoding="utf-8") as f:
    content = f.read()

documents = parse_documents(content)

print(f"Total chunks: {len(documents)}")

# ----------------------------------------
# Show parsed documents
# ----------------------------------------

print("\nDocuments:")

for document in documents:

    print("Title:", document["title"])
    print("Category:", document["category"])
    print("Text:", document["text"][:100])
    print("-" * 60)

# prepare pincone records
records = []

for document_index, document in enumerate(documents):
    chunks = chunk_text(
        document["text"],
        chunk_size = 500,
        chunk_overlap = 100,
    )

    for chunk_index, chunk in enumerate(chunks):
        record = {
            "id": (
                f"document-{document_index}"
                f"-chunk-{chunk_index}"
            ),
            "text": chunk,
            "title": document["title"],
            "category": document["category"],
            "source": "knowledge.txt",
        }

        records.append(record)

print(f"Total records: {len(records)}")

#-----------------------------
# Insert into Pinecone
#-----------------------------
index.upsert_records(
    namespace="knowledge-v5",
    records=records,
)

print("knowledgebase inserted into Pinecone successfully.")
