# GAN Training Report

## 1. Objective

The objective of this project was to implement and train a Generative Adversarial Network capable of generating synthetic handwritten-digit images from random noise.

The experiment uses the MNIST dataset as the real data distribution.

## 2. Model Design

### Generator

The Generator maps a 100-dimensional random latent vector to a 28x28 grayscale image.

The network first projects the latent vector into a 7x7 feature map and then uses transposed convolution layers to increase the spatial resolution to 28x28.

### Discriminator

The Discriminator receives a 28x28 grayscale image and produces a probability indicating whether the image is real or generated.

Convolutional layers extract image features while LeakyReLU and dropout improve training behavior.

## 3. Training Procedure

The training process alternates between two objectives:

1. The Discriminator learns to distinguish real MNIST images from generated images.
2. The Generator learns to produce images that the Discriminator classifies as real.

Binary cross-entropy is used as the adversarial loss.

Adam optimization is used with a learning rate of 0.0002 and beta1 of 0.5.

## 4. Training Dynamics

GAN training is different from ordinary supervised learning because there is no single loss that simply decreases toward zero.

The Generator and Discriminator losses interact with each other. A strong Discriminator can temporarily make the Generator loss increase, while a rapidly improving Generator can make Discriminator classification more difficult.

The generated sample grids saved at each epoch are therefore an important qualitative measure of progress.

## 5. Mode Collapse

Mode collapse is a common GAN failure mode in which the Generator produces very similar images repeatedly instead of covering different patterns in the training distribution.

For MNIST, this could appear as many generated images resembling the same digit or having very similar shapes.

Potential mitigation strategies include:

- Monitoring generated samples throughout training.
- Keeping the Generator and Discriminator learning rates balanced.
- Using Batch Normalization in the Generator.
- Using dropout in the Discriminator.
- Keeping a fixed noise vector so progress can be compared between epochs.
- Avoiding excessively aggressive discriminator updates.

## 6. Results

Generated samples are saved in:

```text
output/generated_samples/
```

The filenames correspond to the training epoch, allowing visual comparison of generation quality over time.

The final model checkpoints are saved in:

```text
output/generator_final.pth
output/discriminator_final.pth
```

The training loss curve is saved as:

```text
output/training_losses.png
```

## 7. Challenges

### Unstable Training
GANs can oscillate because two networks are optimized against each other. Loss values should therefore be interpreted together with generated image quality.

### Generator Quality
Early generated images are usually noisy and lack recognizable structure. More meaningful digit-like forms should emerge as training progresses.

### Mode Collapse
Repeated visual patterns can indicate that the Generator is producing a limited subset of the data distribution. Sample grids across epochs help detect this problem.

## 8. Conclusion

The project demonstrates the fundamental adversarial learning process behind GANs. A Generator learns to synthesize images while a Discriminator learns to identify generated samples.

The experiment also demonstrates why GAN evaluation requires both numerical training metrics and visual inspection of generated samples.
