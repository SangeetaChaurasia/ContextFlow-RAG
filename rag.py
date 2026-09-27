import os

from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not pinecone_api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured")

if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

SIMILARITY_THRESHOLD = 0.30

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("pinecone-semantic-learning")

gemini = genai.Client(api_key=gemini_api_key)

question = "What is the company's vacation policy?"

results = index.search(
    namespace="knowledge-v3",
    inputs={
        "text": question
    },
    top_k=3,
)

print("\nPinecone results:")

for hit in results.result.hits:
    print("ID:", hit.id)
    print("Score:", hit.score)
    print("Text:", hit.fields.get("text"))
    print("-" * 60)

relevant_hits = [
    hit
    for hit in results.result.hits
    if hit.score >= SIMILARITY_THRESHOLD
]

print("\nRelevant results:", len(relevant_hits))

if not relevant_hits:
    print("\nAnswer:")
    print(
        "I don't have enough information in my knowledge base "
        "to answer that."
    )

    exit()

context_parts = []

for hit in results.result.hits:
    text = hit.fields.get("text")
    title = hit.fields.get("title")
    category = hit.fields.get("category")
    source = hit.fields.get("source")
    if text:
        context_parts.append(
            f"""
            Source: {source}
            Title: {title}
            Category: {category}

            Content:
            {text}
            """
        )

context = "\n\n".join(context_parts)

prompt = f"""
You are a helpful knowledgebase assistant.

Your job is to answer the user's question using ONLY the
provided knowledge base context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I don't have enough information in my knowledge base to answer that."
4. Give a concise and clear answer.
5. If possible, mention the source document.

Context:
{context}

Question:
{question}

Answer:
"""

response = gemini.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
)

print("\nQuestion:")
print(question)

print("\nRetrieved Context:")
print(context)

print("\nAnswer:")
print(response.text)