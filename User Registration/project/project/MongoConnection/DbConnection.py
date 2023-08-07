# model/user_model.py

from pymongo import MongoClient
# from bson.objectid import ObjectId
# from Model.schema import UserModel
# from typing import List

class UserModelDb:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["user_database"]
        self.collection = self.db["users"]

    # def registerUsername(self, user: UserModel):
    #     if self.collection.find_one({"username": user.username}):
    #         return False
    #     self.collection.insert_one(user.dict())
    #     return True
    
    # def getRegisteredUsers(self) -> List[str]:
    #     users = self.collection.find({}, {"_id": 0, "username": 1})
    #     return [user["username"] for user in users]
    # def get_user_by_id(self, user_id: str) -> UserModel:
    #     user_data = self.collection.find_one({"_id": ObjectId(user_id)})
    #     if user_data:
    #         return UserModel(**user_data)
    #     else:
    #         return None