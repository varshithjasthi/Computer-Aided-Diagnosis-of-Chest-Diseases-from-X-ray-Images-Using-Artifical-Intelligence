import os
import torch
import timm
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


def main():

    # =====================
    # SETTINGS
    # =====================
    data_dir = r"C:\Users\giriv\OneDrive\Desktop\new x\dataset_split"
    batch_size = 16
    epochs = 10
    lr = 1e-4
    num_workers = 4   # you can keep 4 now
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # TRANSFORMS
    # =====================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # =====================
    # DATASET
    # =====================
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    # =====================
    # MODEL (DeiT)
    # =====================
    model = timm.create_model(
        "deit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )

    model.to(device)

    # =====================
    # LOSS + OPTIMIZER
    # =====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # =====================
    # TRAIN LOOP
    # =====================
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # TEST
        model.eval()
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train Acc:  {train_acc:.4f}")
        print(f"Test Acc:   {test_acc:.4f}")

    torch.save(model.state_dict(), "deit_model.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()
