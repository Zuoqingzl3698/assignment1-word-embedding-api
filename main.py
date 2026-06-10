from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_md")


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV!"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word
    doc = nlp(word)

    return {
        "word": word,
        "embedding": doc.vector.tolist(),
        "embedding_length": len(doc.vector)
    }
