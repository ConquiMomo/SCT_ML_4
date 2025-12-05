from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from src.utils import set_seed

RAW = Path("data/raw/asl_alphabet_train")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")

def build_interim():
    INTERIM.mkdir(parents=True, exist_ok=True)
    for class_dir in RAW.glob("*"):
        if class_dir.is_dir():
            cname = class_dir.name
            (INTERIM / cname).mkdir(parents=True, exist_ok=True)
            for img in class_dir.glob("*.jpg"):
                shutil.copy(img, INTERIM / cname / img.name)

def stratified_split(test_size=0.15, val_size=0.15, seed=42):
    classes = [d.name for d in INTERIM.glob("*") if d.is_dir()]
    for split in ["train", "val", "test"]:
        for cname in classes:
            (PROCESSED / split / cname).mkdir(parents=True, exist_ok=True)

    for cname in classes:
        imgs = list((INTERIM / cname).glob("*.jpg"))
        train_imgs, temp_imgs = train_test_split(imgs, test_size=val_size+test_size, random_state=seed)
        val_ratio = val_size / (val_size + test_size)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1-val_ratio, random_state=seed)

        for p in train_imgs: shutil.copy(p, PROCESSED / "train" / cname / p.name)
        for p in val_imgs: shutil.copy(p, PROCESSED / "val" / cname / p.name)
        for p in test_imgs: shutil.copy(p, PROCESSED / "test" / cname / p.name)

if __name__ == "__main__":
    set_seed(42)
    build_interim()
    stratified_split()
    print("ASL dataset prepared into train/val/test splits")