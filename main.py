from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms

from cnn_model import SimpleCNN
from pydantic import BaseModel
import spacy

app = FastAPI()
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

cnn_model = SimpleCNN()
cnn_model.load_state_dict(
    torch.load("cifar10_cnn.pt", map_location=device)
)
cnn_model.to(device)
cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

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

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    input_tensor = image_transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]

    return {
        "filename": file.filename,
        "predicted_class": predicted_label,
        "confidence": round(confidence.item(), 4)
    }
