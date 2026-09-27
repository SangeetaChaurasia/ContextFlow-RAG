import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sparse_model import encode_sparse

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError(
        "PINECONE_API_KEY is not configured."
    )

pc = Pinecone(api_key=api_key)
index = pc.Index(
    "pinecone-hybrid-learning"
)

namespace = "knowledge-hybrid-v1"

dense_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def search(
    question,
    company_id,
    top_k=5,
    alpha=0.5,
    category=None,
    order_id=None,
):
    if not company_id:
        raise ValueError("company_id is required for tenant isolation.")
        
    dense_vector = dense_model.encode(
        question
    ).tolist()

    sparse_vector = encode_sparse(
        question
    )

    scaled_dense = [
        value * alpha
        for value in dense_vector
    ]

    scaled_sparse = {
        "indices": sparse_vector["indices"],
        "values": [
            value * (1 - alpha)
            for value in sparse_vector["values"]
        ],
    }

    metadata_filter = None

    conditions = []

    if company_id is not None:
        conditions.append({
            "company_id": {
                "$eq": company_id
            }
        })

    if category is not None:
        conditions.append({
            "category": {
                "$eq": category
            }
        })

    if order_id is not None:
        conditions.append({
            "order_id": {
                "$eq": order_id
            }
        })

    if len(conditions) == 1:
        metadata_filter = conditions[0]

    elif len(conditions) > 1:
        metadata_filter = {
            "$and": conditions
        }

    results = index.query(
        namespace=namespace,
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )

    return results
