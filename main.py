from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI(
    title="Word Embedding API",
    description="A simple FastAPI app that returns SpaCy word embeddings.",
    version="1.0"
)

nlp = spacy.load("en_core_web_md")


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def home():
    return {
        "message": "Word Embedding API is running. Go to /docs to test it."
    }


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word
    doc = nlp(word)

    return {
        "query_word": word,
        "embedding": doc.vector.tolist(),
        "embedding_length": len(doc.vector)
    }
