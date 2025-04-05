from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import json
import os
from text_processing import preprocess_text
from compute_cosine import compute_cosine_similarity  
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow requests from all origins (you can restrict this to specific origins if needed)
import nltk

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("tf_idf_matrix.json", "r") as f:
    tf_idf = json.load(f)
with open("inverted_index_matrix.json", "r") as f:
    inverted_index = json.load(f)
with open("fused_image_data.json", "r") as f:
    documents = json.load(f)

@app.get("/search")
async def search(query: str = Query(...)):
    query_tokens = preprocess_text(query)
    if not query_tokens:
        raise HTTPException(status_code=400, detail="Invalid query")
    ranked_docs = compute_cosine_similarity(tf_idf, query_tokens, inverted_index, len(documents))

    results = [
        {
            "doc_id": doc_id,
            "title": documents[int(doc_id)].get("title", ""),
            "description": documents[int(doc_id)].get("ImageDescription", "") or documents[int(doc_id)].get("Artist", ""),
            "image_url": documents[int(doc_id)].get("image_url", "")
        }
        for doc_id, score in ranked_docs[:10]
    ]

    return {"query": query, "results": results}
