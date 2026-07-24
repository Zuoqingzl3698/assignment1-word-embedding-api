import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def offset_cosine_diffusion_schedule(
    diffusion_times: torch.Tensor,
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert diffusion times into noise rates and signal rates.

    diffusion_times should contain values between 0 and 1.

    Returns:
        noise_rates
        signal_rates
    """

    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device,
        )
    )

    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device,
        )
    )

    diffusion_angles = (
        start_angle
        + diffusion_times * (end_angle - start_angle)
    )

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    """
    Encode one diffusion value using sine and cosine functions.

    Input:
        [batch_size, 1, 1, 1]

    Output:
        [batch_size, embedding_dim, 1, 1]
    """

    def __init__(
        self,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                "embedding_dim must be an even number."
            )

        self.embedding_dim = embedding_dim
        number_of_frequencies = embedding_dim // 2

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                number_of_frequencies,
            )
        )

        angular_speeds = (
            2.0
            * math.pi
            * frequencies.view(
                1,
                number_of_frequencies,
                1,
                1,
            )
        )

        self.register_buffer(
            "angular_speeds",
            angular_speeds,
        )

    def forward(
        self,
        diffusion_values: torch.Tensor,
    ) -> torch.Tensor:
        angles = diffusion_values * self.angular_speeds

        return torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )


class ResidualBlock(nn.Module):
    """Residual convolutional block used inside the U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def swish(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return tensor * torch.sigmoid(tensor)

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.projection(tensor)

        tensor = self.conv1(tensor)
        tensor = self.swish(tensor)
        tensor = self.conv2(tensor)

        return tensor + residual


class DownBlock(nn.Module):
    """Residual blocks followed by average-pooling downsampling."""

    def __init__(
        self,
        width: int,
        block_depth: int,
        in_channels: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=width,
                )
            )

            in_channels = width

        self.pool = nn.AvgPool2d(
            kernel_size=2,
        )

    def forward(
        self,
        tensor: torch.Tensor,
        skip_connections: list[torch.Tensor],
    ) -> torch.Tensor:
        for block in self.blocks:
            tensor = block(tensor)
            skip_connections.append(tensor)

        return self.pool(tensor)


class UpBlock(nn.Module):
    """Upsampling followed by residual blocks and skip connections."""

    def __init__(
        self,
        width: int,
        block_depth: int,
        in_channels: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(
                    in_channels=in_channels + width,
                    out_channels=width,
                )
            )

            in_channels = width

    def forward(
        self,
        tensor: torch.Tensor,
        skip_connections: list[torch.Tensor],
    ) -> torch.Tensor:
        tensor = F.interpolate(
            tensor,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        for block in self.blocks:
            skip_tensor = skip_connections.pop()

            tensor = torch.cat(
                [tensor, skip_tensor],
                dim=1,
            )

            tensor = block(tensor)

        return tensor


class UNet(nn.Module):
    """
    U-Net that predicts noise in a noisy CIFAR-10 image.

    Inputs:
        noisy_images:
            [batch_size, 3, 32, 32]

        noise_variances:
            [batch_size, 1, 1, 1]

    Output:
        predicted_noise:
            [batch_size, 3, 32, 32]
    """

    def __init__(
        self,
        image_size: int = 32,
        num_channels: int = 3,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.num_channels = num_channels

        self.initial = nn.Conv2d(
            num_channels,
            32,
            kernel_size=1,
        )

        self.embedding = SinusoidalEmbedding(
            embedding_dim=embedding_dim,
        )

        # Initial image features have 32 channels.
        # Time embedding adds another 32 channels.
        self.down1 = DownBlock(
            width=32,
            block_depth=2,
            in_channels=32 + embedding_dim,
        )

        self.down2 = DownBlock(
            width=64,
            block_depth=2,
            in_channels=32,
        )

        self.down3 = DownBlock(
            width=96,
            block_depth=2,
            in_channels=64,
        )

        self.middle1 = ResidualBlock(
            in_channels=96,
            out_channels=128,
        )

        self.middle2 = ResidualBlock(
            in_channels=128,
            out_channels=128,
        )

        self.up1 = UpBlock(
            width=96,
            block_depth=2,
            in_channels=128,
        )

        self.up2 = UpBlock(
            width=64,
            block_depth=2,
            in_channels=96,
        )

        self.up3 = UpBlock(
            width=32,
            block_depth=2,
            in_channels=64,
        )

        self.final = nn.Conv2d(
            32,
            num_channels,
            kernel_size=1,
        )

        # The course implementation begins with zero noise predictions.
        nn.init.zeros_(self.final.weight)

        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)

    def forward(
        self,
        noisy_images: torch.Tensor,
        noise_variances: torch.Tensor,
    ) -> torch.Tensor:
        skip_connections = []

        image_features = self.initial(noisy_images)

        noise_embedding = self.embedding(
            noise_variances
        )

        noise_embedding = F.interpolate(
            noise_embedding,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        tensor = torch.cat(
            [image_features, noise_embedding],
            dim=1,
        )

        tensor = self.down1(
            tensor,
            skip_connections,
        )

        tensor = self.down2(
            tensor,
            skip_connections,
        )

        tensor = self.down3(
            tensor,
            skip_connections,
        )

        tensor = self.middle1(tensor)
        tensor = self.middle2(tensor)

        tensor = self.up1(
            tensor,
            skip_connections,
        )

        tensor = self.up2(
            tensor,
            skip_connections,
        )

        tensor = self.up3(
            tensor,
            skip_connections,
        )

        return self.final(tensor)


class DiffusionModel(nn.Module):
    """
    Wrapper containing diffusion training and generation logic.
    """

    def __init__(
        self,
        network: UNet,
        schedule_function=offset_cosine_diffusion_schedule,
        ema_decay: float = 0.8,
    ) -> None:
        super().__init__()

        self.network = network
        self.ema_network = copy.deepcopy(network)

        self.ema_network.eval()

        for parameter in self.ema_network.parameters():
            parameter.requires_grad_(False)

        self.schedule_function = schedule_function
        self.ema_decay = ema_decay

        self.register_buffer(
            "normalizer_mean",
            torch.zeros(
                1,
                network.num_channels,
                1,
                1,
            ),
        )

        self.register_buffer(
            "normalizer_std",
            torch.ones(
                1,
                network.num_channels,
                1,
                1,
            ),
        )

    def set_normalizer(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        mean = mean.reshape(
            1,
            self.network.num_channels,
            1,
            1,
        )

        std = std.reshape(
            1,
            self.network.num_channels,
            1,
            1,
        )

        self.normalizer_mean.copy_(
            mean.to(self.normalizer_mean.device)
        )

        self.normalizer_std.copy_(
            std.to(self.normalizer_std.device)
        )

    def normalize(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return (
            images - self.normalizer_mean
        ) / self.normalizer_std.clamp_min(1e-6)

    def denormalize(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        images = (
            images * self.normalizer_std
            + self.normalizer_mean
        )

        return images.clamp(0.0, 1.0)

    def denoise(
        self,
        noisy_images: torch.Tensor,
        noise_rates: torch.Tensor,
        signal_rates: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if training:
            network = self.network
        else:
            network = self.ema_network

        predicted_noises = network(
            noisy_images,
            noise_rates.square(),
        )

        predicted_images = (
            noisy_images
            - noise_rates * predicted_noises
        ) / signal_rates.clamp_min(1e-6)

        return predicted_noises, predicted_images

    @torch.no_grad()
    def update_ema(self) -> None:
        """
        Update the exponential moving average network after training.
        """

        for ema_parameter, parameter in zip(
            self.ema_network.parameters(),
            self.network.parameters(),
        ):
            ema_parameter.mul_(self.ema_decay)

            ema_parameter.add_(
                parameter,
                alpha=1.0 - self.ema_decay,
            )

    def train_step(
        self,
        images: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        self.network.train()

        normalized_images = self.normalize(images)

        true_noises = torch.randn_like(
            normalized_images
        )

        diffusion_times = torch.rand(
            images.size(0),
            1,
            1,
            1,
            device=images.device,
        )

        noise_rates, signal_rates = (
            self.schedule_function(diffusion_times)
        )

        noisy_images = (
            signal_rates * normalized_images
            + noise_rates * true_noises
        )

        predicted_noises, _ = self.denoise(
            noisy_images=noisy_images,
            noise_rates=noise_rates,
            signal_rates=signal_rates,
            training=True,
        )

        loss = loss_function(
            predicted_noises,
            true_noises,
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        self.update_ema()

        return loss.detach().item()

    @torch.no_grad()
    def test_step(
        self,
        images: torch.Tensor,
        loss_function: nn.Module,
    ) -> float:
        self.ema_network.eval()

        normalized_images = self.normalize(images)
        true_noises = torch.randn_like(normalized_images)

        diffusion_times = torch.rand(
            images.size(0),
            1,
            1,
            1,
            device=images.device,
        )

        noise_rates, signal_rates = (
            self.schedule_function(diffusion_times)
        )

        noisy_images = (
            signal_rates * normalized_images
            + noise_rates * true_noises
        )

        predicted_noises, _ = self.denoise(
            noisy_images=noisy_images,
            noise_rates=noise_rates,
            signal_rates=signal_rates,
            training=False,
        )

        loss = loss_function(
            predicted_noises,
            true_noises,
        )

        return loss.item()

    @torch.no_grad()
    def reverse_diffusion(
        self,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
    ) -> torch.Tensor:
        if diffusion_steps < 1:
            raise ValueError(
                "diffusion_steps must be at least 1."
            )

        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        predicted_images = current_images

        for step in range(diffusion_steps):
            diffusion_time = 1.0 - step * step_size

            diffusion_times = torch.full(
                (
                    initial_noise.size(0),
                    1,
                    1,
                    1,
                ),
                diffusion_time,
                device=initial_noise.device,
            )

            noise_rates, signal_rates = (
                self.schedule_function(diffusion_times)
            )

            predicted_noises, predicted_images = (
                self.denoise(
                    noisy_images=current_images,
                    noise_rates=noise_rates,
                    signal_rates=signal_rates,
                    training=False,
                )
            )

            next_diffusion_times = torch.clamp(
                diffusion_times - step_size,
                min=0.0,
            )

            next_noise_rates, next_signal_rates = (
                self.schedule_function(
                    next_diffusion_times
                )
            )

            current_images = (
                next_signal_rates * predicted_images
                + next_noise_rates * predicted_noises
            )

        return predicted_images

    @torch.no_grad()
    def generate(
        self,
        num_images: int,
        diffusion_steps: int = 50,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(
            self.network.parameters()
        ).device

        if initial_noise is None:
            initial_noise = torch.randn(
                num_images,
                self.network.num_channels,
                self.network.image_size,
                self.network.image_size,
                device=device,
            )

        generated_images = self.reverse_diffusion(
            initial_noise=initial_noise,
            diffusion_steps=diffusion_steps,
        )

        return self.denormalize(
            generated_images
        )