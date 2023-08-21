from fastapi import FastAPI
from Routes.routes import router

app = FastAPI()

# Include the router
app.include_router(router)
