from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API opérationnelle"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"prediction": "stub", "labels": ["Chihuahua", "Beagle", "Golden"]}
