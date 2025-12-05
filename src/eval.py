import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from src.utils import set_seed, get_device

DATA = Path("data/processed")
MODELS = Path("models")
OUTPUTS = Path("outputs"); OUTPUTS.mkdir(exist_ok=True)

def get_eval_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    transform_eval = get_eval_transform()

    # Try multiple possible dataset locations
    test_path = None
    for candidate in [DATA / "test", DATA / "val", Path("data/raw/asl_alphabet_test")]:
        if candidate.exists():
            test_path = candidate
            break

    if test_path is None:
        raise FileNotFoundError("No test/val dataset found. Please create data/processed/test or point to the correct folder.")

    print(f"Using evaluation dataset at: {test_path}")

    test_ds = datasets.ImageFolder(test_path, transform=transform_eval)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    checkpoint = torch.load(MODELS / "best_model.pt", map_location=device)
    classes = checkpoint["classes"]

    from src.train import SmallCNN
    model = SmallCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Save metrics
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    with open(OUTPUTS / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    plt.colorbar(im)
    plt.title("Confusion Matrix")
    plt.savefig(OUTPUTS / "confusion_matrix.png")
    plt.close()

    print("Evaluation complete. Metrics and confusion matrix saved.")