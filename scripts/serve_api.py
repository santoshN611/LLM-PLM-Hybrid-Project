#!/usr/bin/env python3
from fastapi import FastAPI
import uvicorn
from rag_pipeline import answer

print("🚀 Launching RAG‐API…")
app = FastAPI()

@app.get('/qa')
def qa(q: str):
    resp = answer(q)
    return {'answer': resp}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
