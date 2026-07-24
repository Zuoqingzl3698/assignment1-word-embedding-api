from pathlib import Path

from torchvision import datasets, transforms
from torchvision.utils import save_image


def main():
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    images = []
    labels = []

    for index in range(16):
        image, label = dataset[index]
        images.append(image)
        labels.append(label)

    import torch

    image_batch = torch.stack(images)

    print("Number of CIFAR-10 training images:", len(dataset))
    print("Image batch shape:", image_batch.shape)
    print("Labels:", labels)

    save_image(
        image_batch,
        "outputs/cifar10_preview.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )

    print("Saved preview to outputs/cifar10_preview.png")


if __name__ == "__main__":
    main()