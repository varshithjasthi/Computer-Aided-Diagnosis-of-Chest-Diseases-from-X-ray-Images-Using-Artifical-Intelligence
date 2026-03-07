# ==============================
# Install dependency
# ==============================
!pip install timm -q

# ==============================
# Imports
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
from tqdm import tqdm

# ==============================
# Config
# ==============================
DATASET_PATH = "/kaggle/input/xray-dataset/dataset_15"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ==============================
# Augmentations
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ==============================
# Dataset
# ==============================
train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH,"train"),
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# ==============================
# Swin Model
# ==============================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=num_classes
)

model = model.to(DEVICE)

# ==============================
# Loss + Optimizer
# ==============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.cuda.amp.GradScaler()

# ==============================
# Training Loop
# ==============================
for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs,1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_acc = 100 * correct / total

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Accuracy: {train_acc:.2f}%")

# ==============================
# Save Model
# ==============================
torch.save({
    "model": model.state_dict(),
    "class_names": class_names
}, "swin_ultra_fast_15_classes.pth")

print("Model saved!")
