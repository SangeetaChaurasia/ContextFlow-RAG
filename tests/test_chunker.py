from utils.chunker import chunk_test

text = """
Customers can request a refund within 30 days of purchase.

To request a refund, open the Orders section of your account.

Select the order you want to refund and click the Request Refund button.

Refund requests are reviewed within 2 business days.

Approved refunds are returned to the original payment method.
"""

chunks = chunk_test(
    text,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print(f"Chunk {index + 1}: {chunk}")
    print(chunk)