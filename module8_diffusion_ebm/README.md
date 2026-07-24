# CIFAR-10 Diffusion and Energy-Based Model API

This project extends the FastAPI application developed in an earlier class activity by adding two CIFAR-10 image generators:

- An Energy-Based Model
- A Diffusion Model

The application also retains the existing word-embedding, CIFAR-10 classification, and MNIST GAN endpoints.

## Energy-Based Model

The Energy-Based Model uses a convolutional neural network to assign a scalar energy value to each image.

Real CIFAR-10 images are trained to have lower energy, while generated negative samples are trained to have higher energy.

Negative samples are generated using Langevin Dynamics. During sampling, gradients are computed with respect to the input images rather than the model parameters.

## Diffusion Model

The Diffusion Model uses a U-Net with sinusoidal embeddings.

During training, Gaussian noise is added to CIFAR-10 images. The U-Net learns to predict the added noise.

During generation, the model starts from random Gaussian noise and applies reverse diffusion to produce an image.

## Dataset

Both models use the CIFAR-10 training dataset. CIFAR-10 contains 50,000 training images in 10 categories.

The dataset is downloaded automatically using torchvision.

## Project Files

- `main.py`: FastAPI application
- `energy_model.py`: Energy network, Langevin sampling, and replay buffer
- `train_ebm.py`: EBM training script
- `diffusion_model.py`: U-Net and diffusion generation methods
- `train_diffusion.py`: Diffusion training script
- `cnn_model.py`: CIFAR-10 classification network
- `gan_model.py`: MNIST GAN generator
- `check_cifar10.py`: CIFAR-10 data-loading test

## Installation

Create and activate an Anaconda environment:

```bash
conda create -n module8 python=3.11 -y
conda activate module8
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy model:

```bash
python -m spacy download en_core_web_md
```

## Model Training

Train the Energy-Based Model:

```bash
python train_ebm.py --epochs 10 --batch-size 32 --langevin-steps 10
```

Train the Diffusion Model:

```bash
python train_diffusion.py --epochs 2 --batch-size 16
```

The checkpoints are saved in the `models` directory.

## Run the API

```bash
uvicorn main:app --reload
```

Open the interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `GET /`
- `POST /embedding`
- `POST /classify-image`
- `GET /generate-digits`
- `GET /generate-energy`
- `GET /generate-diffusion`

## Gradient Handling

During Langevin sampling, gradients are computed with respect to the input image:

```python
images.requires_grad_(True)

image_gradients, = torch.autograd.grad(
    outputs=energies,
    inputs=images,
    grad_outputs=torch.ones_like(energies),
)
```

The gradients modify the input images toward lower-energy states.

During model training, `loss.backward()` and `optimizer.step()` are used to update the neural-network parameters instead.