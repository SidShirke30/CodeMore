import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from gan_models import Generator, Discriminator


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_SAMPLE_IMAGES = 64

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "generated_samples")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


def save_sample_grid(generator, fixed_noise, epoch, device):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise.to(device)).cpu()

    path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png")
    save_image(
        fake_images,
        path,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )
    generator.train()
    return path


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)

    initialize_weights(generator)
    initialize_weights(discriminator)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )

    fixed_noise = torch.randn(NUM_SAMPLE_IMAGES, LATENT_DIM)
    g_losses = []
    d_losses = []

    print(f"Training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # -------------------------
            # Train Discriminator
            # -------------------------
            optimizer_d.zero_grad()

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            real_output = discriminator(real_images)
            d_real_loss = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise)

            fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()

            # -------------------------
            # Train Generator
            # -------------------------
            optimizer_g.zero_grad()

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            generated_images = generator(noise)
            output = discriminator(generated_images)

            # Generator wants fake images to be classified as real.
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_g.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d_loss = running_d_loss / len(loader)
        avg_g_loss = running_g_loss / len(loader)

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        sample_path = save_sample_grid(
            generator, fixed_noise, epoch, device
        )

        torch.save(
            generator.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch:03d}.pth"),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                f"discriminator_epoch_{epoch:03d}.pth",
            ),
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"D Loss: {avg_d_loss:.4f} | "
            f"G Loss: {avg_g_loss:.4f} | "
            f"Sample: {sample_path}"
        )

    # Save final models.
    torch.save(
        generator.state_dict(),
        os.path.join(OUTPUT_DIR, "generator_final.pth"),
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(OUTPUT_DIR, "discriminator_final.pth"),
    )

    # Save loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "training_losses.png"),
        dpi=150,
    )
    plt.close()

    print("\nTraining complete.")
    print(f"Generated samples: {SAMPLE_DIR}")
    print(f"Models and plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
