from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms

from cnn_model import SimpleCNN
from pydantic import BaseModel
import spacy
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from torchvision.utils import save_image

from gan_model import Generator
from energy_model import EnergyModel, generate_samples

from diffusion_model import (
    DiffusionModel,
    UNet,
    offset_cosine_diffusion_schedule,
)

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

BASE_DIR = Path(__file__).resolve().parent

energy_model = EnergyModel().to(device)

energy_checkpoint_path = (
    BASE_DIR / "models" / "cifar10_ebm.pth"
)

energy_checkpoint = torch.load(
    energy_checkpoint_path,
    map_location=device,
)

energy_model.load_state_dict(
    energy_checkpoint["model_state_dict"]
)

energy_model.eval()

diffusion_network = UNet(
    image_size=32,
    num_channels=3,
    embedding_dim=32,
)

diffusion_generator = DiffusionModel(
    network=diffusion_network,
    schedule_function=offset_cosine_diffusion_schedule,
    ema_decay=0.8,
).to(device)

diffusion_checkpoint_path = (
    BASE_DIR / "models" / "cifar10_diffusion.pth"
)

diffusion_checkpoint = torch.load(
    diffusion_checkpoint_path,
    map_location=device,
)

diffusion_generator.network.load_state_dict(
    diffusion_checkpoint["network_state_dict"]
)

diffusion_generator.ema_network.load_state_dict(
    diffusion_checkpoint["ema_network_state_dict"]
)

diffusion_generator.set_normalizer(
    mean=diffusion_checkpoint[
        "normalizer_mean"
    ].reshape(-1),
    std=diffusion_checkpoint[
        "normalizer_std"
    ].reshape(-1),
)

diffusion_generator.network.eval()
diffusion_generator.ema_network.eval()

LATENT_DIM = 100

mnist_generator = Generator(
    latent_dim=LATENT_DIM
)

generator_model_path = (
    BASE_DIR / "models" / "mnist_generator.pth"
)

mnist_generator.load_state_dict(
    torch.load(
        generator_model_path,
        map_location="cpu"
    )
)

mnist_generator.eval()

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

@app.get("/generate-digits")
def generate_digits(num_images: int = 16):
    if num_images < 1 or num_images > 64:
        raise HTTPException(
            status_code=400,
            detail="num_images must be between 1 and 64"
        )

    noise = torch.randn(
        num_images,
        LATENT_DIM
    )

    with torch.no_grad():
        generated_images = mnist_generator(
            noise
        )

    image_buffer = BytesIO()

    save_image(
        generated_images,
        image_buffer,
        format="PNG",
        nrow=min(8, num_images),
        normalize=True,
        value_range=(-1, 1)
    )

    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png"
    )

@app.get("/generate-energy")
def generate_energy(
    num_images: int = 16,
    sampling_steps: int = 20,
):
    if num_images < 1 or num_images > 64:
        raise HTTPException(
            status_code=400,
            detail="num_images must be between 1 and 64",
        )

    if sampling_steps < 1 or sampling_steps > 200:
        raise HTTPException(
            status_code=400,
            detail="sampling_steps must be between 1 and 200",
        )

    initial_images = (
        torch.rand(
            num_images,
            3,
            32,
            32,
            device=device,
        )
        * 2
        - 1
    )

    # EBM generation requires gradients with respect
    # to the input images.
    with torch.enable_grad():
        generated_images = generate_samples(
            energy_model=energy_model,
            input_images=initial_images,
            steps=sampling_steps,
            step_size=10.0,
            noise_std=0.01,
        )

    image_buffer = BytesIO()

    save_image(
        generated_images.cpu(),
        image_buffer,
        format="PNG",
        nrow=min(8, num_images),
        normalize=True,
        value_range=(-1, 1),
    )

    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png",
    )

@app.get("/generate-diffusion")
def generate_diffusion(
    num_images: int = 16,
    diffusion_steps: int = 20,
):
    if num_images < 1 or num_images > 64:
        raise HTTPException(
            status_code=400,
            detail="num_images must be between 1 and 64",
        )

    if diffusion_steps < 1 or diffusion_steps > 200:
        raise HTTPException(
            status_code=400,
            detail="diffusion_steps must be between 1 and 200",
        )

    with torch.no_grad():
        generated_images = diffusion_generator.generate(
            num_images=num_images,
            diffusion_steps=diffusion_steps,
        )

    image_buffer = BytesIO()

    save_image(
        generated_images.cpu(),
        image_buffer,
        format="PNG",
        nrow=min(8, num_images),
    )

    image_buffer.seek(0)

    return StreamingResponse(
        image_buffer,
        media_type="image/png",
    )