from fastapi import FastAPI
#from fastapi.templating import Jinja2Templates
from Routes.user_routes import router as user_router
import uvicorn

app = FastAPI()
#templates = Jinja2Templates(directory="templates")

app.include_router(user_router)
def start():
 uvicorn.run("main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    

# def start():
  uvicorn.run("main:app", host="127.0.0.1", port=8000) 