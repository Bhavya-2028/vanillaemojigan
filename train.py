# ============================================================
# MODULE 0: IMPORTS & CONFIGURATION
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 28
LATENT_DIM = 20
BATCH_SIZE = 32
EPOCHS = 300
LR = 0.0002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# ============================================================
# MODULE 1: RAW DATASET + DATA PIPELINE
# ============================================================
class SmileyDataset(Dataset):
    def __init__(self, samples=500):
        self.samples = samples

    def draw_smiley(self):
        img = np.zeros((IMG_SIZE, IMG_SIZE))

        # Face (circle)
        cx, cy, r = 14, 14, 10
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    img[x, y] = 1

        # Eyes
        img[10, 10] = 0
        img[10, 18] = 0

        # Smile
        for x in range(10, 19):
            img[18, x] = 0

        return img

    def __getitem__(self, idx):
        img = self.draw_smiley()
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img * 2 - 1  # Normalize [-1, 1]

    def __len__(self):
        return self.samples

dataset = SmileyDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# MODULE 2: MODEL DESIGN (VANILLA GAN)
# ============================================================

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        return self.generator(z).view(-1, 1, IMG_SIZE, IMG_SIZE)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.discriminator(img.view(img.size(0), -1))

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# ============================================================
# MODULE 3: GAN TRAINING SETUP
# ============================================================
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# ============================================================
# MODULE 4: GAN TRAINING PROCESS
# ============================================================
for epoch in range(EPOCHS):
    for real_imgs in loader:
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.9
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # Train Discriminator
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)

        d_loss = criterion(D(real_imgs), real_labels) + \
                 criterion(D(fake_imgs.detach()), fake_labels)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{EPOCHS}")

# ============================================================
# MODULE 5: EVALUATION & QUALITY ANALYSIS (MULTIPLE OUTPUTS)
# ============================================================
with torch.no_grad():
    z = torch.randn(9, LATENT_DIM, device=DEVICE)   # generate 9 emojis
    generated_imgs = (G(z).cpu() + 1) / 2

plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(generated_imgs[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.suptitle("Generated Smiling Emojis (Vanilla GAN)")
plt.show()

# ============================================================
# MODULE 6: DEPLOYMENT (CONCEPT)
# ============================================================
# Trained generator can be saved and reused for emoji generation

# ============================================================
# MODULE 7: MONITORING & UPDATE (CONCEPT)
# ============================================================
# Generator and Discriminator losses monitored during training

