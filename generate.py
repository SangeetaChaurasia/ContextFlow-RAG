import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai(api_key=api_key)

question = "How long is a password reset link valid?"

context = """
The password reset link expires after 30 minutes.

Enter the email address associated with your account.
You will receive an email containing a password reset link.

To reset your password, go to the login page and click the
"Forgot password" link.
"""

prompt = f"""
Answer the user's question using ONLY the provided context.

context:
{context}

question:
{question}

answer:
"""

response = client.models.generate_content(
    model="gemini-3.7-flash",
    input=prompt,
)

print(response.output_text)