from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.utils import set_seed, get_device

DATA = Path("data/processed")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

def get_transforms():
    mean, std = [0.5], [0.5]
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform_train, transform_eval

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return epoch_loss / total, correct / total

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    transform_train, transform_eval = get_transforms()

    train_ds = datasets.ImageFolder(DATA / "train", transform=transform_train)
    val_ds   = datasets.ImageFolder(DATA / "val",   transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    num_classes = len(train_ds.classes)
    model = SmallCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.0
    for epoch in range(10):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch+1}: train_acc={ta:.3f} val_acc={va:.3f}")
        if va > best_val:
            best_val = va
            torch.save({"model_state": model.state_dict(), "classes": train_ds.classes}, MODELS / "best_model.pt")