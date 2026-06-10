#Assignment1
FastAPI app for returning a SpaCy word embedding.
#Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn main:app --reload
