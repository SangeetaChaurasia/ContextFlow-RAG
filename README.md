# ContextFlow — Production-Style Hybrid RAG API

ContextFlow is a production-style Retrieval-Augmented Generation (RAG) API built with **Python, FastAPI, Pinecone, hybrid dense+sparse retrieval, cross-encoder reranking, Gemini, JWT authentication, conversational query rewriting, metadata filtering, and incremental indexing**.

The project is designed as a portfolio-quality example of how a modern RAG system can be structured beyond a simple vector-search demo.

---

# Project Overview

Traditional RAG systems commonly perform:

```
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
LLM
   ↓
Answer
```

ContextFlow extends this pipeline:

```
                         ┌────────────────────┐
                         │      Client        │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │     FastAPI        │
                         │  JWT Authentication│
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Conversation Store │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Query Rewriter     │
                         │ Gemini             │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Query Classifier   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Hybrid Retrieval            │
                    │ Dense + Sparse              │
                    │ Metadata Filters             │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Cross-Encoder Reranking     │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Context Builder              │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Gemini Generation            │
                    │ Grounded Answer              │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                              Response
```

---

# Key Features

## 1. Hybrid Search

Uses both:

- Dense semantic embeddings
- Sparse lexical embeddings

This improves retrieval for both natural-language questions and exact terms such as:

```
ORD-83921
TRK-552781
password reset
refund
```

---

## 2. Pinecone Vector Database

Pinecone is used for scalable vector storage and retrieval.

The hybrid index contains:

- Dense vectors
- Sparse vectors
- Document metadata

---

## 3. Cross-Encoder Reranking

Initial retrieval returns a larger candidate set.

A cross-encoder then scores:

```
Question + Document
```

together and selects the most relevant context.

This helps when several documents are semantically similar.

---

## 4. Conversational RAG

The API supports follow-up questions.

Example:

```
User:
What is the status of ORD-83921?

Assistant:
The order is currently in transit.

User:
When was it shipped?

Assistant:
It was shipped on August 15, 2026.
```

The second question is rewritten internally into a standalone question containing the required context.

---

## 5. Query Classification

Queries are classified into categories such as:

```
order
shipping
account
refund
general
```

The classification can be used to apply category-specific retrieval filters.

---

## 6. Entity Extraction

Order IDs are detected from questions:

```
ORD-83921
```

When an order ID is detected, it is used as a Pinecone metadata filter.

This prevents a reranker from selecting a different order with similar wording.

---

## 7. Multi-Tenant Retrieval

Every authenticated user contains:

```
user_id
company_id
```

The `company_id` is used as a mandatory metadata filter for `/ask`.

This prevents one tenant from retrieving another tenant's knowledge.

---

## 8. Incremental Indexing

The ingestion pipeline tracks document hashes and chunk IDs.

On subsequent runs:

```
Unchanged document → Skip
Changed document   → Re-index
New document       → Index
Deleted document   → Remove stale chunks
```

This avoids rebuilding the entire Pinecone namespace unnecessarily.

---

## 9. JWT Authentication

The `/ask` endpoint requires a Bearer token.

The JWT contains:

```
{
  "user_id": "...",
  "company_id": "...",
  "exp": 1234567890
}
```

---

## 10. Request Tracing

Each request receives a unique request ID.

Example log:

```
[request-id] Query rewritten in 0.421s
[request-id] Extracted order_id=ORD-83921
[request-id] Query classified as: order
```

This makes debugging and latency analysis easier.

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Application runtime |
| FastAPI | REST API |
| Pydantic | Request validation |
| Pinecone | Vector database |
| Sentence Transformers | Dense embeddings / reranking |
| SPLADE / Transformers | Sparse retrieval |
| Google Gemini | Query rewriting and answer generation |
| PyJWT | Authentication |
| Pytest | Automated testing |
| dotenv | Environment configuration |


# Prerequisites

Install the following before starting:

- Python 3.10+
- Git
- A Pinecone account
- A Google AI / Gemini API key

Verify Python:

```
python3 --version
```

Example:

```
Python 3.10.x
```

---

# Installation

## 1. Clone the repository

```
git clone <repository-url>
cd contextflow-rag-api
```

Replace `<repository-url>` with the GitHub repository URL after publishing the project.

---

## 2. Create a virtual environment

```
python3 -m venv .venv
```

---

## 3. Activate the virtual environment

Linux/macOS:

```
source .venv/bin/activate
```

Windows:

```
.venv\Scripts\activate
```

You should see something similar to:

```
(.venv) user@machine:~/contextflow-rag-api$
```

---

## 4. Upgrade pip

```
python -m pip install --upgrade pip
```

---

## 5. Install dependencies

```
pip install -r requirements.txt
```

Verify the environment:

```
pip check
```

Expected:

```
No broken requirements found.
```

---

# Environment Configuration

Create the environment file:

```
cp .env.example .env
```

Edit it:

```
nano .env
```

Example:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=pinecone-hybrid-learning
PINECONE_NAMESPACE=knowledge-hybrid-v1

GEMINI_API_KEY=your_gemini_api_key

JWT_SECRET=replace_with_a_long_random_secret
```

# Prepare Knowledge Base

The sample knowledge base is located at:

```
data/knowledge.txt
```

It contains example support information such as:

```
Password Reset
Refund Policy
Order Tracking
Change Email Address
Order ORD-83921
Order ORD-74125
```

The order examples demonstrate exact entity retrieval.

Example:

```
ORD-83921
TRK-552781
```

---

# Create Pinecone Index

ContextFlow uses a hybrid Pinecone index.

The project uses:

```
Dense dimension: 384
Metric: dotproduct
```

The dense representation is generated using:

```
SentenceTransformer("all-MiniLM-L6-v2")
```

The sparse representation is generated using a SPLADE-style model.

The exact Pinecone configuration should match the configuration implemented in the project's index setup module.

---

# Build the Hybrid Index

Run the ingestion pipeline using the project's ingestion script.

For example:

```
python hybrid_ingest.py
```

The ingestion process:

```
knowledge.txt
     ↓
Document parsing
     ↓
Chunking
     ↓
Document hashing
     ↓
Dense embedding
     ↓
Sparse embedding
     ↓
Metadata generation
     ↓
Pinecone upsert
     ↓
Manifest update
```

---

# Incremental Ingestion

The ingestion pipeline maintains:

```
data/chunk_manifest.json
data/document_manifest.json
```

Document hashes are used to detect changes.

### First run

```
6 documents
→ 6 documents indexed
```

### Second run without changes

```
6 documents
→ 0 documents re-indexed
```

### If one document changes

```
Changed document
→ regenerate affected chunks
→ upsert updated vectors
```

### If a document is deleted

```
Deleted document
→ identify stale chunks
→ delete them from Pinecone
```

This makes ingestion suitable for recurring indexing workflows.

---

# Run the API

Start the FastAPI application:

```
uvicorn api:app --reload
```

The API will normally be available at:

```
http://127.0.0.1:8000
```

---

# API Documentation

FastAPI automatically provides Swagger documentation:

```
http://127.0.0.1:8000/docs
```

OpenAPI JSON:

```
http://127.0.0.1:8000/openapi.json
```

Swagger UI can be used to test the endpoints directly.

---

# Authentication

Generate a development JWT:

```
python generate_token.py
```

The generated token contains:

```
user_id
company_id
expiration
```

Example payload:

```
{
  "user_id": "user-123",
  "company_id": "company-1",
  "exp": 1234567890
}
```

Use it as:

```
Authorization: Bearer <JWT_TOKEN>
```

Do not commit real tokens to GitHub.

---

# API Endpoints

## GET /

Health/basic API endpoint.

```
GET /
```

---

## POST /search

Performs retrieval without generating a final LLM answer.

Typical request:

```
{
  "question": "What is the status of ORD-83921?",
  "company_id": "company-1",
  "top_k": 5,
  "alpha": 0.5
}
```

---

## POST /ask

Runs the complete RAG pipeline.

Typical request:

```
{
  "question": "What is the status of ORD-83921?",
  "session_id": "demo-session",
  "top_k": 5,
  "alpha": 0.5
}
```

The company ID is taken from the authenticated JWT rather than trusted from the request body.

---

# Search API

Example using curl:

```
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the status of ORD-83921?",
    "company_id": "company-1",
    "top_k": 5,
    "alpha": 0.5
  }'
```

The retrieval pipeline performs:

```
Question
   ↓
Dense embedding
   +
Sparse embedding
   ↓
Pinecone hybrid search
   ↓
Metadata filtering
   ↓
Candidate documents
   ↓
Reranking
   ↓
Relevant context
```

---

# Conversational RAG API

`/ask` supports session-based conversations.

Example:

```
{
  "question": "What is the status of ORD-83921?",
  "session_id": "session-001"
}
```

Then:

```
{
  "question": "When was it shipped?",
  "session_id": "session-001"
}
```

The second question is rewritten using conversation history.

Conceptually:

```
"When was it shipped?"
```

becomes:

```
"When was order ORD-83921 shipped?"
```

The rewritten question is then passed through retrieval.

---

# Conversation Example

### Request 1

```
What is the status of ORD-83921?
```

The system retrieves the order document.

---

### Request 2

```
When was it shipped?
```

Conversation history tells the system that:

```
"it" = ORD-83921
```

The query rewriter creates a standalone question.

The order ID is extracted:

```
ORD-83921
```

The metadata filter is applied.

The correct order document is retrieved.

---

# How Retrieval Works

The retrieval pipeline can be summarized as:

```
User Question
      ↓
Conversation History
      ↓
Query Rewriting
      ↓
Entity Extraction
      ↓
Query Classification
      ↓
Metadata Filter
      ↓
Dense + Sparse Retrieval
      ↓
Candidate Results
      ↓
Cross-Encoder Reranking
      ↓
Context Builder
      ↓
Gemini
      ↓
Grounded Answer
```

---

# Hybrid Retrieval

Hybrid retrieval combines two signals.

## Dense Retrieval

Dense embeddings capture semantic meaning.

For example:

```
"Where is my package?"
```

can match:

```
"Your shipment is currently in transit."
```

even though the words are different.

---

## Sparse Retrieval

Sparse retrieval is particularly useful for exact terms.

For example:

```
ORD-83921
TRK-552781
```

Exact identifiers can be difficult for purely semantic retrieval.

---

## Alpha

The hybrid search uses an `alpha` value to balance dense and sparse retrieval.

Conceptually:

```
alpha → dense importance
1-alpha → sparse importance
```

Example:

```
alpha = 0.5
```

gives balanced weighting.

The API allows this value to be tuned during experimentation.

---

# Metadata Filtering and Multi-Tenancy

Documents contain metadata such as:

```
{
  "company_id": "company-1",
  "category": "orders",
  "order_id": "ORD-83921"
}
```

The retrieval filter can combine:

```
company_id
category
order_id
```

Example:

```
company_id = company-1
AND
category = orders
AND
order_id = ORD-83921
```

This is important for SaaS-style applications where multiple companies share the same vector infrastructure.

---

# Reranking

Pinecone returns the initial candidate set.

For example:

```
Top 5 candidates
```

The cross-encoder then evaluates:

```
(question, document)
```

pairs.

The final context contains the highest-ranked documents.

This two-stage design is:

```
Fast retrieval
      ↓
Candidate generation
      ↓
More expensive reranking
      ↓
High-quality context
```

This is generally more efficient than applying an expensive reranker to the entire corpus.

---

# Grounded Generation

The generator receives retrieved context rather than relying only on the model's general knowledge.

Conceptually:

```
System instructions
+
Retrieved context
+
User question
        ↓
      Gemini
        ↓
Grounded answer
```

The goal is to reduce hallucination by making the retrieved knowledge the basis for the response.

The project uses:

```
Gemini 3.8 Flash
```

for generation and query rewriting.

---

# Unanswerable Questions

The knowledge base does not contain every possible detail.

For example, if an order document contains:

```
Status: shipped
Shipping date: August 15, 2026
Tracking: TRK-552781
```

but does not contain a delivery date, the system should not invent one.

A good RAG system must distinguish between:

```
Information found
```

and:

```
Information not present in the knowledge base
```

The evaluation dataset includes an unanswerable example for this purpose.

---

# Evaluation

The project includes:

```
evaluation_dataset.json
evaluate_rag.py
```

The evaluation dataset contains answerable and unanswerable questions.

Example categories include:

```
Order status
Shipping date
Tracking number
Password reset
Refund policy
Order tracking
Unanswerable delivery-date question
```

Run:

```
python evaluate_rag.py
```

The evaluation script uses the configured evaluation company:

```
company-1
```

The goal is to measure retrieval quality and detect regressions when retrieval logic changes.

---

# Automated Tests

Tests are located in:

```
tests/
```

Run all tests:

```
python -m pytest
```

Run a specific test file:

```
python -m pytest tests/test_classifier.py
```

Run with verbose output:

```
python -m pytest -v
```

Example classifier tests cover:

```
Order question
Account question
Refund question
Shipping question
General question
```

The retrieval tests should also verify tenant isolation and relevant filtering behavior.

---

# Logging and Error Handling

The API generates request IDs to make individual requests traceable.

Example:

```
[request-id] Query rewritten in 0.421s
[request-id] Extracted order_id=ORD-83921
[request-id] Query classified as: order
```

Timing information is useful for identifying expensive stages such as:

```
Query rewriting
Embedding
Pinecone retrieval
Reranking
Generation
```

External model/API operations use retry and backoff where appropriate.

API failures are logged and returned as controlled HTTP errors rather than exposing raw internal exceptions.

---

# Security Notes

This repository is intended as a portfolio/learning project, but several production security practices are demonstrated.

## Never commit secrets

Do not commit:

```
.env
PINECONE_API_KEY
GEMINI_API_KEY
JWT_SECRET
real JWT tokens
```

The `.gitignore` should include:

```
.env
.venv/
__pycache__/
*.pyc
```

---

## JWT Secret

For local learning, a simple secret can be used.

For production, generate a long unpredictable secret.

Example:

```
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## Token Expiration

JWTs contain an expiration time.

Expired tokens should not be accepted.

---

## Tenant Isolation

The `/ask` endpoint obtains `company_id` from the authenticated JWT.

Do not trust a client-provided company ID for authorization.

---

# Troubleshooting

## `ModuleNotFoundError`

Make sure the virtual environment is active:

```
source .venv/bin/activate
```

Then:

```
python -m pip install -r requirements.txt
```

---

## `pip check` reports dependency conflicts

Run:

```
pip check
```

Then inspect:

```
pip freeze
```

Resolve version conflicts before continuing.

---

## Pinecone authentication error

Check:

```env
PINECONE_API_KEY=...
```

Make sure the key is valid and the correct Pinecone project is being used.

---

## Gemini authentication error

Check:

```env
GEMINI_API_KEY=...
```

Do not place the key directly in source code.

---

## No retrieval results

Check:

1. Pinecone index name
2. Namespace
3. `company_id`
4. Metadata filters
5. Whether ingestion completed
6. Whether vectors exist in the expected namespace

---

## Wrong order retrieved

Check whether the question contains an order ID:

```text
ORD-83921
```

The application extracts the ID and uses it as a metadata filter.

This is especially important when multiple order documents contain similar language.

---

## Follow-up question loses context

Check that:

```text
user question
```

is added to conversation history **after** retrieval and generation, not before.

Otherwise the current question can incorrectly appear in the history used to rewrite itself.

---

# Design Decisions

## Why Pinecone?

Pinecone provides managed vector search suitable for production-style applications without requiring the application to manage vector database infrastructure.

---

## Why Hybrid Search?

Dense search is strong for semantic similarity.

Sparse search is strong for exact words and identifiers.

Combining both provides a better retrieval strategy for support-style data.

---

## Why Rerank?

Initial vector retrieval is optimized for speed.

Cross-encoder reranking improves precision on a smaller candidate set.

---

## Why Query Rewriting?

Users naturally ask follow-up questions using pronouns:

```text
What is the status of ORD-83921?

When was it shipped?
```

Query rewriting converts conversational questions into standalone retrieval queries.

---

## Why Metadata Filtering?

Metadata filters provide deterministic constraints.

For example:

```text
order_id = ORD-83921
```

is stronger than hoping semantic similarity ranks the correct order first.

---

## Why Incremental Indexing?

Re-embedding the entire knowledge base whenever one document changes is inefficient.

Hash-based change detection allows only affected documents to be processed.

---

# Future Improvements

Possible future production enhancements include:

- Persistent conversation storage
- Redis-backed sessions
- PostgreSQL for application data
- Background ingestion workers
- Job queues
- Streaming responses
- Advanced evaluation metrics
- Retrieval observability dashboard
- Rate limiting
- API key management
- Role-based access control
- Production deployment with Docker
- CI/CD
- Automated evaluation in CI
- Document-level access control
- Better citation/source presentation

These are intentionally left as future improvements rather than adding unnecessary infrastructure to the core project.

---

# Resume Description

### Short Version

**ContextFlow — Production-Style Hybrid RAG API**

Built a multi-tenant RAG API using FastAPI and Pinecone with hybrid dense+sparse retrieval, cross-encoder reranking, conversational query rewriting, metadata filtering, JWT authentication, incremental indexing, grounded Gemini generation, automated evaluation, and request-level observability.

### Resume Bullet Points

- Built a production-style **Hybrid RAG API** using FastAPI, Pinecone, dense embeddings, SPLADE sparse retrieval, and cross-encoder reranking.
- Implemented **multi-tenant retrieval isolation** using JWT-based authentication and Pinecone metadata filtering.
- Added **conversational RAG** with LLM-based query rewriting for context-aware follow-up questions.
- Implemented **entity-aware retrieval** for exact identifiers such as order and tracking IDs.
- Developed **incremental document ingestion** using SHA-256 document hashing and stable chunk IDs to avoid unnecessary re-indexing.
- Added **evaluation datasets, automated tests, retry/backoff, structured request tracing, and latency logging**.
- Integrated **Gemini 3.8 Flash** for grounded answer generation and conversational query rewriting.

---

# GitHub Publishing Checklist

Before making the repository public, run:

```
git status
```

Confirm:

```text
.env is NOT tracked
.venv is NOT tracked
API keys are NOT present
JWT tokens are NOT present
```

Check tracked files:

```
git ls-files
```

Search the project for common secret patterns:

```
grep -R "PINECONE_API_KEY" . \
  --exclude-dir=.git \
  --exclude-dir=.venv
```

Also inspect:

```
git diff
git log --all --oneline
```

If a secret was ever committed to Git history, simply deleting it from the latest file is not sufficient. Rotate the secret and clean the Git history before publishing.

---

# Typical Development Workflow

After cloning:

```
cd contextflow-rag-api

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

cp .env.example .env
```

Configure `.env`.

Then:

```
python hybrid_ingest.py
```

Run tests:

```
python -m pytest
```

Run evaluation:

```
python evaluate_rag.py
```

Generate a development token:

```
python generate_token.py
```

Start the API:

```
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

Then test:

```
POST /search
POST /ask
```

---

# End-to-End Flow

The complete project workflow is:

```
                    KNOWLEDGE BASE
                          │
                          ▼
                    Document Parser
                          │
                          ▼
                        Chunker
                          │
                          ▼
                  Change Detection
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Dense Embedding           Sparse Embedding
             │                         │
             └────────────┬────────────┘
                          ▼
                     Pinecone
                          │
                          │
                    USER QUESTION
                          │
                          ▼
                    JWT Validation
                          │
                          ▼
                  Conversation History
                          │
                          ▼
                    Query Rewriting
                          │
                          ▼
                  Entity Extraction
                          │
                          ▼
                  Query Classification
                          │
                          ▼
                  Metadata Filtering
                          │
                          ▼
                Hybrid Pinecone Search
                          │
                          ▼
                    Cross Encoder
                      Reranking
                          │
                          ▼
                   Context Builder
                          │
                          ▼
                  Gemini 3.8 Flash
                          │
                          ▼
                  Grounded Response
```

---

# Conclusion

ContextFlow demonstrates the major components required to move from a basic vector-search experiment toward a more realistic RAG application:

- Hybrid retrieval
- Semantic and lexical search
- Reranking
- Conversational retrieval
- Query classification
- Entity-aware filtering
- Multi-tenant isolation
- Incremental indexing
- JWT authentication
- Grounded generation
- Evaluation
- Automated testing
- Logging and retry handling

The project is intentionally modular so individual components can be replaced or scaled independently.
