"""
Assignment 1a: Convolutional Neural Network (CNN) on Fashion-MNIST (PyTorch)

Run:  python cnn_fashion_mnist.py [--epochs 8]
Outputs (in ./outputs): cnn_curves.png, cnn_confusion_matrix.png, cnn_samples.png,
                        cnn_results.txt, cnn_model.pt
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class CNN(nn.Module):
    """Two conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool) + fully connected head."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)                               # halves H and W
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 32 x 14 x 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 64 x 7 x 7
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_set = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    # Sample images
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    for ax, i in zip(axes.flat, range(16)):
        img, lbl = train_set[i]
        ax.imshow(img.squeeze() * 0.3530 + 0.2860, cmap="gray")
        ax.set_title(CLASSES[lbl], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/cnn_samples.png", dpi=120)
    plt.close()

    model = CNN().to(device)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"tr_loss": [], "tr_acc": [], "te_loss": [], "te_acc": []}
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, device, criterion)
        for k, v in zip(hist, (tr_loss, tr_acc, te_loss, te_acc)):
            hist[k].append(v)
        print(f"Epoch {ep:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| test loss {te_loss:.4f} acc {te_acc:.4f}")

    # Curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep_axis = range(1, args.epochs + 1)
    ax[0].plot(ep_axis, hist["tr_loss"], label="train"); ax[0].plot(ep_axis, hist["te_loss"], label="test")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep_axis, hist["tr_acc"], label="train"); ax[1].plot(ep_axis, hist["te_acc"], label="test")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig("outputs/cnn_curves.png", dpi=120); plt.close()

    # Final evaluation
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im)
    plt.tight_layout(); plt.savefig("outputs/cnn_confusion_matrix.png", dpi=120); plt.close()

    report = classification_report(labels, preds, target_names=CLASSES, digits=4)
    summary = (f"CNN on Fashion-MNIST\nEpochs: {args.epochs}\n"
               f"Final test accuracy: {hist['te_acc'][-1]:.4f}\n"
               f"Final test loss: {hist['te_loss'][-1]:.4f}\n\n{report}")
    print("\n" + summary)
    with open("outputs/cnn_results.txt", "w") as f:
        f.write(summary)
    torch.save(model.state_dict(), "outputs/cnn_model.pt")


if __name__ == "__main__":
    main()
