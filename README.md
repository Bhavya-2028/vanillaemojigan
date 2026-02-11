# 🎭 Vanilla Emoji GAN

A Generative Adversarial Network (GAN) implemented using PyTorch to generate synthetic smiling emoji images.

---

## 📌 Project Overview

This project implements a **Vanilla GAN** architecture consisting of:

- Generator Network
- Discriminator Network
- Binary Cross Entropy Loss
- Adam Optimizer

The model is trained to generate 28x28 grayscale smiling emoji images from random noise vectors.

---

## 🧠 Model Architecture

### Generator
- Fully Connected Layers
- ReLU Activation
- Tanh Output Layer
- Input: 20-dimensional latent vector

### Discriminator
- Fully Connected Layers
- LeakyReLU Activation
- Sigmoid Output Layer
- Output: Real / Fake probability

---

## ⚙️ Training Details

- Image Size: 28x28
- Latent Dimension: 20
- Batch Size: 32
- Epochs: 300
- Learning Rate: 0.0002
- Framework: PyTorch
- Environment: Google Colab

---

## 📊 Output

After training, the model generates multiple synthetic smiling emoji images from random latent vectors.

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
python train.py
```

---

## 🔮 Future Improvements

- Upgrade to DCGAN using convolutional layers
- Train on a real emoji dataset
- Save trained model weights (.pth file)
- Deploy using Streamlit web interface

---




