# rag_app/main.py
from app.config import QDRANT_PATH
from fastapi import FastAPI, Query
from app.query import query_engine
from app.ingest import ingest_documents
import os 

app = FastAPI()

@app.on_event("startup")
def load_index():
    ingest_documents()


@app.get("/")
def read_root():
    return {"message": "RAG API is running. Use /ask?q=your_question to query."}

@app.get("/ask")
def ask_question(query: str = Query(..., description="Enter your question")):
    answer = query_engine(query)
    return {"answer": answer}
