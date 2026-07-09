import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import os

from models import Generator, Discriminator


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

noise_dim = 100
batch_size = 128
epochs = 5
lr = 0.0002

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading MNIST dataset...")

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

small_dataset = Subset(dataset, range(10000))
dataloader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)

print("Dataset loaded. Start training...")

generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

os.makedirs("samples", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    for batch_index, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        optimizer_d.zero_grad()

        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        noise = torch.randn(current_batch_size, noise_dim).to(device)
        fake_images = generator(noise)

        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()

        noise = torch.randn(current_batch_size, noise_dim).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Batch [{batch_index + 1}/{len(dataloader)}], "
            f"D Loss: {d_loss.item()}, "
            f"G Loss: {g_loss.item()}"
        )

    sample_noise = torch.randn(16, noise_dim).to(device)
    sample_images = generator(sample_noise)
    save_image(sample_images, f"samples/epoch_{epoch + 1}.png", normalize=True)

torch.save(generator.state_dict(), "checkpoints/gan_generator.pth")
torch.save(discriminator.state_dict(), "checkpoints/gan_discriminator.pth")

print("Training finished.")
print("Saved generator to checkpoints/gan_generator.pth")
print("Saved discriminator to checkpoints/gan_discriminator.pth")
print("Saved sample images to samples folder")