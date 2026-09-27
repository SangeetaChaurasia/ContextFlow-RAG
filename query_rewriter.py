import os

from google import genai
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured"
    )


client = genai.Client(
    api_key=api_key
)


def rewrite_question(question, history):
    """
    Convert a follow-up question into a
    standalone question using conversation history.
    """

    if not history:
        return question

    conversation = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in history
    )

    prompt = f"""
You are a question rewriting assistant.

Convert the user's latest question into
a standalone question that can be searched
in a knowledge base.

Use the conversation history to resolve
references such as:

- it
- its
- they
- that order
- this issue
- the previous request

Do not answer the question.

Return ONLY the rewritten question.

CONVERSATION HISTORY:
{conversation}

LATEST USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    return response.text.strip()
