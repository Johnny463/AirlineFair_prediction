#Controller/user.py

from MongoConnection.DbConnection import UserModelDb
from Model.schema import UserModel
from typing import List
from bson.objectid import ObjectId

model = UserModelDb()

class UserController:
    def __init__(self):
        self.model = model

    async def registerUsername(self, user: UserModel) -> object:
        if self.model.collection.find_one(user.username):
            return {"message": "User registered successfully", "is_registered": True}
        self.model.collection.insert_one(user.dict())
        return {"message": "User registered successfully", "is_registered": True}

    def getRegisteredUsers(self) -> List[str]:
        users = self.model.collection.find({}, {"_id": 0, "username": 1})
        return [user["username"] for user in users]

    def get_user_by_id(self, user_id: str) -> UserModel:
        user_data = self.model.collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return UserModel(**user_data)
        else:
            return None
