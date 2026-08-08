# Week 6 - Task 2: Generative Adversarial Network (GAN)

## Overview
This project implements a Generative Adversarial Network (GAN) using PyTorch and the MNIST handwritten-digit dataset. The GAN contains two competing neural networks:

- **Generator:** converts random noise vectors into synthetic 28x28 grayscale images.
- **Discriminator:** classifies images as real MNIST images or generated/fake images.

The networks are trained adversarially so that the Generator gradually learns to create images that look like handwritten digits.

## Project Structure

```text
Task 2/
├── README.md
├── requirements.txt
├── src/
│   ├── gan_models.py
│   └── train_gan.py
├── output/
│   └── generated_samples/
└── docs/
    └── gan_report.md
```

## Dataset
The project uses **MNIST**, which contains 28x28 grayscale handwritten-digit images.

The dataset is automatically downloaded by torchvision the first time the training script is run.

## GAN Architecture

### Generator
The Generator receives a random latent vector of size 100 and progressively transforms it into a 28x28 image using transposed convolution layers.

Architecture:
- Linear projection
- Batch normalization
- Transposed convolution
- ReLU activations
- Final Tanh activation

### Discriminator
The Discriminator receives a 28x28 image and predicts whether it is real or generated.

Architecture:
- Convolution layers
- LeakyReLU activations
- Dropout
- Linear output
- Sigmoid probability

## Training
The GAN uses binary cross-entropy loss and the Adam optimizer.

Default hyperparameters:

| Parameter | Value |
|---|---:|
| Image size | 28x28 |
| Latent dimension | 100 |
| Batch size | 128 |
| Epochs | 20 |
| Learning rate | 0.0002 |
| Adam beta1 | 0.5 |
| Adam beta2 | 0.999 |

For every batch:
1. Train the Discriminator using real images.
2. Train the Discriminator using generated images.
3. Generate new images.
4. Train the Generator so the Discriminator classifies its images as real.
5. Save generated samples at regular intervals.

## Installation

Create and activate a virtual environment if desired, then install:

```bash
pip install -r requirements.txt
```

## Run

From the `Task 2` directory:

```bash
python src/train_gan.py
```

Generated samples will be saved under:

```text
output/generated_samples/
```

Model checkpoints are saved under:

```text
output/checkpoints/
```

## Results
During training, generated images should gradually develop recognizable digit-like structures. Early samples are generally noisy, while later samples should become more coherent.

See `docs/gan_report.md` for the discussion of training dynamics, challenges, mode collapse, and results.

## Notes
GPU acceleration is used automatically when CUDA is available. Otherwise, training runs on CPU.
