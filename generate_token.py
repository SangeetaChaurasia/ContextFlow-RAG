import os
import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

secret = os.getenv("JWT_SECRET")

payload = {
    "user_id": "user-123",
    "company_id": "company-1",
    "exp": datetime.now(timezone.utc) + timedelta(hours=1),
}

token = jwt.encode(
    payload,
    secret,
    algorithm="HS256",
)

print(token)
