import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from diffusion_model import (
    DiffusionModel,
    UNet,
    offset_cosine_diffusion_schedule,
)


CIFAR10_MEAN = torch.tensor([
    0.4914,
    0.4822,
    0.4465,
])

CIFAR10_STD = torch.tensor([
    0.2470,
    0.2435,
    0.2616,
])


def get_device() -> torch.device:
    """Use Apple GPU, NVIDIA GPU, or CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_cifar10_loader(
    batch_size: int,
) -> DataLoader:
    """Load CIFAR-10 images in the range [0, 1]."""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


def save_generated_images(
    diffusion_model: DiffusionModel,
    epoch: int,
    diffusion_steps: int,
) -> None:
    """Generate and save a CIFAR-10 image grid."""

    diffusion_model.eval()

    generated_images = diffusion_model.generate(
        num_images=16,
        diffusion_steps=diffusion_steps,
    )

    output_path = Path(
        f"outputs/diffusion_epoch_{epoch:02d}.png"
    )

    save_image(
        generated_images.cpu(),
        output_path,
        nrow=4,
    )

    print(f"Saved generated images to {output_path}")


def save_checkpoint(
    diffusion_model: DiffusionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Save both the training network and EMA network."""

    checkpoint_path = Path(
        "models/cifar10_diffusion.pth"
    )

    checkpoint = {
        "epoch": epoch,
        "network_state_dict":
            diffusion_model.network.state_dict(),
        "ema_network_state_dict":
            diffusion_model.ema_network.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "normalizer_mean":
            diffusion_model.normalizer_mean.cpu(),
        "normalizer_std":
            diffusion_model.normalizer_std.cpu(),
        "image_size": 32,
        "num_channels": 3,
        "embedding_dim": 32,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}")


def train(args: argparse.Namespace) -> None:
    """Train the CIFAR-10 diffusion model."""

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    device = get_device()

    print(f"Using device: {device}")

    train_loader = create_cifar10_loader(
        batch_size=args.batch_size,
    )

    network = UNet(
        image_size=32,
        num_channels=3,
        embedding_dim=32,
    )

    diffusion_model = DiffusionModel(
        network=network,
        schedule_function=offset_cosine_diffusion_schedule,
        ema_decay=0.8,
    ).to(device)

    diffusion_model.set_normalizer(
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )

    optimizer = torch.optim.AdamW(
        diffusion_model.network.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    # The course-style implementation predicts the true noise.
    loss_function = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        diffusion_model.train()

        epoch_loss = 0.0
        processed_batches = 0

        for batch_index, (images, _) in enumerate(
            train_loader
        ):
            images = images.to(device)

            loss = diffusion_model.train_step(
                images=images,
                optimizer=optimizer,
                loss_function=loss_function,
            )

            epoch_loss += loss
            processed_batches += 1

            if batch_index % 20 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_index} | "
                    f"Noise loss: {loss:.4f}"
                )

            if (
                args.max_batches is not None
                and processed_batches >= args.max_batches
            ):
                break

        average_loss = (
            epoch_loss / max(processed_batches, 1)
        )

        print(
            f"\nEpoch {epoch} finished | "
            f"Average noise loss: {average_loss:.4f}\n"
        )

        save_checkpoint(
            diffusion_model=diffusion_model,
            optimizer=optimizer,
            epoch=epoch,
        )

        #save_generated_images(
        #    diffusion_model=diffusion_model,
        #    epoch=epoch,
        #    diffusion_steps=args.sample_steps,
       # )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CIFAR-10 diffusion model."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--sample-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_arguments())