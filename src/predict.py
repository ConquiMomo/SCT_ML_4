import argparse
import random
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import get_device, set_seed
from src.train import SmallCNN

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATASET = ROOT / "data" / "raw" / "asl_alphabet_train"

# Device
device = get_device()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load model
checkpoint = torch.load(MODELS / "best_model.pt", map_location=device)
classes = checkpoint["classes"]
model = SmallCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

def predict_image(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)
        pred = out.argmax(1).item()
        label = classes[pred]

    return img, label

def predict_random_batch(n=10):
    # Collect all class folders
    class_folders = [d for d in DATASET.iterdir() if d.is_dir()]
    images = []

    for _ in range(n):
        class_folder = random.choice(class_folders)
        img_path = random.choice(list(class_folder.glob("*.jpg")))
        img, label = predict_image(img_path)
        images.append((img, label))

    # Plot grid
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 6))
    for i, (img, label) in enumerate(images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(f"{label}", fontsize=12, color="green")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--random", action="store_true", help="Pick a random image")
    parser.add_argument("--batch", action="store_true", help="Pick 10 random images")
    args = parser.parse_args()

    if args.image:
        img, label = predict_image(Path(args.image))
        plt.imshow(img)
        plt.title(f"Predicted: {label}", fontsize=16, color="green")
        plt.axis("off")
        plt.show()
    elif args.random:
        img, label = predict_image(random.choice(list(DATASET.rglob("*.jpg"))))
        plt.imshow(img)
        plt.title(f"Predicted: {label}", fontsize=16, color="green")
        plt.axis("off")
        plt.show()
    elif args.batch:
        predict_random_batch(10)
    else:
        print("Please provide --image path/to/file.jpg OR --random OR --batch")