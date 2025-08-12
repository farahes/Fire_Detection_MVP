import os, sys
from pathlib import Path

# Make 'src' importable whether we run with -m or as a file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.config import SPLITS_ROOT

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- hyperparams
    sz = 224
    bs = 32
    epochs = 8
    lr = 3e-4
    NUM_WORKERS = 0  # Windows-safe (avoid multiprocessing "spawn" issues)

    # ----- transforms
    tfm_train = T.Compose([
        T.Resize((sz, sz)),
        T.ColorJitter(0.25, 0.25, 0.25, 0.10),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    tfm_eval = T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
    ])

    # ----- datasets / loaders
    train_dir = SPLITS_ROOT / "train"
    val_dir   = SPLITS_ROOT / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected data_splits at {SPLITS_ROOT}. Run:  python -m scripts.split"
        )

    tr = tv.datasets.ImageFolder(train_dir, transform=tfm_train)
    va = tv.datasets.ImageFolder(val_dir,   transform=tfm_eval)

    tr_loader = DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=NUM_WORKERS)
    va_loader = DataLoader(va, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)

    # ----- model
    model = tv.models.mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # where to save best weights
    best_path = PROJECT_ROOT / "cls_best.pth"
    best = 0.0

    # ----- training loop
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        # validate
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                tot += y.numel()

        acc = correct / tot if tot else 0.0
        train_loss = running / len(tr) if len(tr) else 0.0
        print(f"epoch {ep}: train_loss {train_loss:.4f} | val acc {acc:.3f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), best_path)

    print("best val acc:", best)
    print(f"saved best classifier to: {best_path}")

if __name__ == "__main__":
    main()
