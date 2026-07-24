import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from energy_model import EBM, EnergyModel, generate_samples


def get_device() -> torch.device:
    """Select Apple GPU, NVIDIA GPU, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_cifar10_loader(
    batch_size: int,
) -> DataLoader:
    """Create the CIFAR-10 training data loader."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

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
    model: EnergyModel,
    device: torch.device,
    epoch: int,
    sampling_steps: int,
) -> None:
    """Generate and save an image grid after an epoch."""

    model.eval()

    initial_images = (
        torch.rand(
            16,
            3,
            32,
            32,
            device=device,
        )
        * 2
        - 1
    )

    generated_images = generate_samples(
        energy_model=model,
        input_images=initial_images,
        steps=sampling_steps,
        step_size=10.0,
        noise_std=0.01,
    )

    output_path = Path(
        f"outputs/ebm_epoch_{epoch:02d}.png"
    )

    save_image(
        generated_images,
        output_path,
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )

    print(f"Saved samples to {output_path}")


def train(args: argparse.Namespace) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader = create_cifar10_loader(
        batch_size=args.batch_size
    )

    energy_network = EnergyModel().to(device)

    ebm = EBM(
        model=energy_network,
        alpha=0.1,
        steps=args.langevin_steps,
        step_size=10.0,
        noise_std=0.005,
        device=device,
    )

    optimizer = torch.optim.Adam(
        energy_network.parameters(),
        lr=0.0001,
        betas=(0.0, 0.999),
    )

    for epoch in range(1, args.epochs + 1):
        ebm.reset_metrics()

        for batch_index, (real_images, _) in enumerate(
            train_loader
        ):
            real_images = real_images.to(device)

            metrics = ebm.train_step(
                real_images=real_images,
                optimizer=optimizer,
            )

            if batch_index % 20 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_index} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Real energy: "
                    f"{metrics['real_energy']:.4f} | "
                    f"Fake energy: "
                    f"{metrics['fake_energy']:.4f}"
                )

            # Used for a quick test of only a few batches.
            if (
                args.max_batches is not None
                and batch_index + 1 >= args.max_batches
            ):
                break

        epoch_metrics = ebm.metrics()

        print(
            f"\nEpoch {epoch} finished | "
            f"Loss: {epoch_metrics['loss']:.4f} | "
            f"Contrastive: "
            f"{epoch_metrics['contrastive']:.4f} | "
            f"Regularization: "
            f"{epoch_metrics['regularization']:.4f}\n"
        )

        checkpoint = {
            "model_state_dict": energy_network.state_dict(),
            "epoch": epoch,
            "image_channels": 3,
            "image_size": 32,
        }

        checkpoint_path = Path(
            "models/cifar10_ebm.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Saved checkpoint to {checkpoint_path}"
        )

        save_generated_images(
            model=energy_network,
            device=device,
            epoch=epoch,
            sampling_steps=args.sample_steps,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 energy-based model."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--langevin-steps",
        type=int,
        default=10,
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