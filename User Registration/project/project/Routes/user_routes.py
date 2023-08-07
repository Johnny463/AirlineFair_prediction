# routes/user_routes.py

from fastapi import APIRouter, HTTPException, Path
from Controller.controller import UserController
from Model.schema import UserModel


router = APIRouter()
user_controller = UserController()

@router.post("/register/", response_model=dict)
async def register(user: UserModel):
    is_registered = user_controller.registerUsername(user)
    return await is_registered

@router.get("/users/", response_model=list[str])
def getUsers():
    return user_controller.getRegisteredUsers()
@router.get("/users/{user_id}", response_model=UserModel)
def getUserById(user_id: str = Path(..., title="User ID")):
    user = user_controller.get_user_by_id(user_id)
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")


