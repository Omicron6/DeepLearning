"""
Assignment 1b: Recurrent Neural Network (LSTM and GRU) on sequential MNIST (PyTorch)

Each 28x28 digit image is read as a sequence of 28 time steps, one row of 28 pixels per step.
The final hidden state is classified into one of 10 digits. LSTM and GRU are trained and compared.

Run:  python rnn_lstm_mnist.py [--epochs 6]
Outputs (in ./outputs): rnn_curves.png, rnn_confusion_matrix.png, rnn_results.txt
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class RNNClassifier(nn.Module):
    def __init__(self, cell="lstm", input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):                 # x: (batch, 1, 28, 28)
        x = x.squeeze(1)                  # -> (batch, 28 time steps, 28 features)
        out, _ = self.rnn(x)              # -> (batch, 28, hidden)
        return self.fc(out[:, -1, :])     # last time step -> class scores


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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilises RNN training
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_model(cell, train_loader, test_loader, device, epochs, lr):
    model = RNNClassifier(cell).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {cell.upper()} ({n_params} trainable parameters) ===")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"tr_loss": [], "tr_acc": [], "te_loss": [], "te_acc": []}
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, device, criterion)
        for k, v in zip(hist, (tr_loss, tr_acc, te_loss, te_acc)):
            hist[k].append(v)
        print(f"Epoch {ep:2d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| test loss {te_loss:.4f} acc {te_acc:.4f}")
    return model, hist, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    results = {}
    for cell in ("lstm", "gru"):
        results[cell] = train_model(cell, train_loader, test_loader, device, args.epochs, args.lr)

    # Curves: LSTM vs GRU
    ep_axis = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cell, (_, h, _) in results.items():
        ax[0].plot(ep_axis, h["tr_loss"], label=f"{cell.upper()} train")
        ax[0].plot(ep_axis, h["te_loss"], "--", label=f"{cell.upper()} test")
        ax[1].plot(ep_axis, h["tr_acc"], label=f"{cell.upper()} train")
        ax[1].plot(ep_axis, h["te_acc"], "--", label=f"{cell.upper()} test")
    ax[0].set_title("Loss"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Epoch"); a.legend()
    plt.tight_layout(); plt.savefig("outputs/rnn_curves.png", dpi=120); plt.close()

    # Confusion matrix for LSTM
    model = results["lstm"][0]
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("LSTM confusion matrix (test set)")
    fig.colorbar(im)
    plt.tight_layout(); plt.savefig("outputs/rnn_confusion_matrix.png", dpi=120); plt.close()

    lines = ["RNN on sequential MNIST (28 steps x 28 features)", f"Epochs: {args.epochs}", ""]
    for cell, (_, h, n_params) in results.items():
        lines.append(f"{cell.upper()}: params={n_params}, final test acc={h['te_acc'][-1]:.4f}, "
                     f"final test loss={h['te_loss'][-1]:.4f}")
    lines += ["", "LSTM classification report:", classification_report(labels, preds, digits=4)]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open("outputs/rnn_results.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
