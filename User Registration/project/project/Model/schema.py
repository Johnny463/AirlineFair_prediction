# model/schema.py

from pydantic import BaseModel

class UserModel(BaseModel):
    username: str #= Field(..., min_length=3, max_length=20, pattern=r"^[a-zA-Z0-9_-]+$")
