import random

import numpy as np
import torch
from torch import nn


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function used in the course example."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """
    CNN that maps a CIFAR-10 image to one scalar energy value.

    Input shape:
        [batch_size, 3, 32, 32]

    Output shape:
        [batch_size, 1]
    """

    def __init__(self) -> None:
        super().__init__()

        # CIFAR-10 has 3 color channels rather than 1 MNIST channel.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.flatten = nn.Flatten()

        # After four stride-2 convolutions:
        # 32 -> 16 -> 8 -> 4 -> 2
        self.fc1 = nn.Linear(
            64 * 2 * 2,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))

        x = self.flatten(x)
        x = swish(self.fc1(x))

        return self.fc2(x)


def generate_samples(
    energy_model: nn.Module,
    input_images: torch.Tensor,
    steps: int,
    step_size: float,
    noise_std: float,
) -> torch.Tensor:
    """
    Use Langevin Dynamics to modify images toward lower-energy states.

    Gradients are calculated with respect to input_images rather than
    being used to update the model parameters.
    """

    energy_model.eval()
    images = input_images.detach()

    for _ in range(steps):
        # Add a small amount of random noise.
        with torch.no_grad():
            noise = torch.randn_like(images) * noise_std
            images = (images + noise).clamp(-1.0, 1.0)

        # Tell PyTorch to calculate gradients with respect to the images.
        images.requires_grad_(True)

        energies = energy_model(images)

        image_gradients, = torch.autograd.grad(
            outputs=energies,
            inputs=images,
            grad_outputs=torch.ones_like(energies),
            create_graph=False,
        )

        # Update the images, not the model parameters.
        with torch.no_grad():
            image_gradients = image_gradients.clamp(
                min=-0.03,
                max=0.03,
            )

            images = (
                images - step_size * image_gradients
            ).clamp(-1.0, 1.0)

        # Disconnect this iteration from the next computation graph.
        images = images.detach()

    return images


class Buffer:
    """
    Replay buffer containing previously generated images.

    Most samples begin from old buffer images. A small percentage begin
    from newly generated random noise.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        initial_size: int = 128,
        maximum_size: int = 8192,
    ) -> None:
        self.model = model
        self.device = device
        self.maximum_size = maximum_size

        self.examples = [
            torch.rand(
                1,
                3,
                32,
                32,
                device=self.device,
            ) * 2 - 1
            for _ in range(initial_size)
        ]

    def sample_new_examples(
        self,
        batch_size: int,
        steps: int,
        step_size: float,
        noise_std: float,
    ) -> torch.Tensor:
        # Approximately 5% of samples start from fresh random noise.
        number_new = np.random.binomial(
            batch_size,
            0.05,
        )

        number_old = batch_size - number_new

        image_groups = []

        if number_new > 0:
            new_random_images = torch.rand(
                number_new,
                3,
                32,
                32,
                device=self.device,
            ) * 2 - 1

            image_groups.append(new_random_images)

        if number_old > 0:
            old_images = torch.cat(
                random.choices(
                    self.examples,
                    k=number_old,
                ),
                dim=0,
            )

            image_groups.append(old_images)

        input_images = torch.cat(
            image_groups,
            dim=0,
        )

        new_images = generate_samples(
            energy_model=self.model,
            input_images=input_images,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std,
        )

        # Put the most recent images at the front of the buffer.
        self.examples = (
            list(torch.split(new_images.detach(), 1, dim=0))
            + self.examples
        )

        self.examples = self.examples[:self.maximum_size]

        return new_images


class Metric:
    """Keep a running average of a metric during one epoch."""

    def __init__(self) -> None:
        self.reset()

    def update(self, value: torch.Tensor) -> None:
        self.total += value.detach().item()
        self.count += 1

    def result(self) -> float:
        if self.count == 0:
            return 0.0

        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class EBM:
    """
    Wrapper responsible for training the energy model.

    Training lowers the energy of real CIFAR-10 images and raises the
    energy of generated negative samples.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float,
        steps: int,
        step_size: float,
        noise_std: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device

        self.buffer = Buffer(
            model=self.model,
            device=self.device,
        )

        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise_std = noise_std

        self.loss_metric = Metric()
        self.regularization_metric = Metric()
        self.contrastive_metric = Metric()
        self.real_energy_metric = Metric()
        self.fake_energy_metric = Metric()

    def reset_metrics(self) -> None:
        self.loss_metric.reset()
        self.regularization_metric.reset()
        self.contrastive_metric.reset()
        self.real_energy_metric.reset()
        self.fake_energy_metric.reset()

    def metrics(self) -> dict[str, float]:
        return {
            "loss": self.loss_metric.result(),
            "regularization": self.regularization_metric.result(),
            "contrastive": self.contrastive_metric.result(),
            "real_energy": self.real_energy_metric.result(),
            "fake_energy": self.fake_energy_metric.result(),
        }

    def train_step(
        self,
        real_images: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        self.model.train()

        batch_size = real_images.size(0)

        # Slightly perturb the real images.
        real_images = (
            real_images
            + torch.randn_like(real_images) * self.noise_std
        ).clamp(-1.0, 1.0)

        # Generate negative samples using Langevin Dynamics.
        fake_images = self.buffer.sample_new_examples(
            batch_size=batch_size,
            steps=self.steps,
            step_size=self.step_size,
            noise_std=self.noise_std,
        )

        self.model.train()

        all_images = torch.cat(
            [real_images, fake_images.detach()],
            dim=0,
        )

        energy_scores = self.model(all_images)

        real_energy, fake_energy = torch.split(
            energy_scores,
            [batch_size, batch_size],
            dim=0,
        )

        # Lower real-image energy and increase fake-image energy.
        contrastive_loss = (
            real_energy.mean()
            - fake_energy.mean()
        )

        # Prevent the energy values from growing without limit.
        regularization_loss = self.alpha * (
            real_energy.pow(2).mean()
            + fake_energy.pow(2).mean()
        )

        loss = contrastive_loss + regularization_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=0.1,
        )

        optimizer.step()

        self.loss_metric.update(loss)
        self.regularization_metric.update(regularization_loss)
        self.contrastive_metric.update(contrastive_loss)
        self.real_energy_metric.update(real_energy.mean())
        self.fake_energy_metric.update(fake_energy.mean())

        return self.metrics()