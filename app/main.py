<<<<<<< HEAD
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

=======
>>>>>>> 49609e01e978a0f529218ad92dc868f66d5ef6e9
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import body_measurements

app = FastAPI(
    title="Virtual Fitting Room API",
    description="API for virtual try-on functionality",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(body_measurements.router, prefix="/api/v1", tags=["body-measurements"])

@app.get("/")
async def root():
    return FileResponse("app/static/index.html") 