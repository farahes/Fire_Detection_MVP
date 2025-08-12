import os, random, shutil
from pathlib import Path
from src.config import TRAINVAL_DIR, TEST_DIR, SPLITS_ROOT, CLASSES, IMG_EXTS

random.seed(7)

def ensure_dirs(root):
    for split in ["train","val","test"]:
        for c in CLASSES:
            (root / split / c).mkdir(parents=True, exist_ok=True)

def collect_images(folder: Path):
    files=[]
    for ext in IMG_EXTS:
        files += list(folder.glob(f"*{ext}"))
        files += list(folder.glob(f"*{ext.upper()}"))
    return files

def main():
    ensure_dirs(SPLITS_ROOT)

    # ---- Train/Val from "Training and Validation"
    for c in CLASSES:
        src = TRAINVAL_DIR / c
        imgs = collect_images(src)
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n*0.8)   # 80/20 split for train/val
        train, val = imgs[:n_train], imgs[n_train:]

        for p in train:
            shutil.copy2(p, SPLITS_ROOT/"train"/c/p.name)
        for p in val:
            shutil.copy2(p, SPLITS_ROOT/"val"/c/p.name)

    # ---- Test from "Testing"
    for c in CLASSES:
        src = TEST_DIR / c
        imgs = collect_images(src)
        for p in imgs:
            shutil.copy2(p, SPLITS_ROOT/"test"/c/p.name)

    print("Done. Created data_splits/{train,val,test}/{fire,nofire}")

if __name__ == "__main__":
    main()
