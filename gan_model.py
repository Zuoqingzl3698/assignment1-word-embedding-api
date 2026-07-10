import torch
from torch import nn


class Generator(nn.Module):
    """
    Input:
        Noise vector: (batch_size, 100)

    Output:
        Generated MNIST image: (batch_size, 1, 28, 28)
    """

    def __init__(self, latent_dim: int = 100):
        super().__init__()

        self.latent_dim = latent_dim

        # 100 -> 7 * 7 * 128
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        # 128 x 7 x 7 -> 64 x 14 x 14
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # 64 x 14 x 14 -> 1 x 28 x 28
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.fc(noise)

        # Reshape to batch_size x 128 x 7 x 7
        x = x.view(-1, 128, 7, 7)

        x = self.deconv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    """
    Input:
        MNIST image: (batch_size, 1, 28, 28)

    Output:
        Real/fake probability: (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        # 1 x 28 x 28 -> 64 x 14 x 14
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.leaky_relu1 = nn.LeakyReLU(
            negative_slope=0.2
        )

        # 64 x 14 x 14 -> 128 x 7 x 7
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batch_norm = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(
            negative_slope=0.2
        )

        self.flatten = nn.Flatten()

        # 128 * 7 * 7 -> 1
        self.fc = nn.Linear(128 * 7 * 7, 1)

        # Convert output to probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv1(image)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.leaky_relu2(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
