import os

from google import genai
from dotenv import load_dotenv
from retry import retry_with_backoff

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )

client = genai.Client(
    api_key=api_key
)

def generate_answer(
    question,
    context,
):
    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the context, say:

"I don't have enough information to answer that."

Do not invent information.

CONTEXT:
{context}

USER QUESTION:
{question}

Answer clearly and concisely.
"""

    def call_gemini():
        return client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )

    response = retry_with_backoff(
        call_gemini,
        max_attempts=3,
        initial_delay=1,
    )

    return response.text
