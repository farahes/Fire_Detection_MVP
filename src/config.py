from pathlib import Path

# Root of the unzipped Kaggle dataset
DATA_ROOT = Path("data")
TRAINVAL_DIR = DATA_ROOT / "Training and Validation"
TEST_DIR     = DATA_ROOT / "Testing"

# Where our pipeline will write splits and masks
SPLITS_ROOT = Path("data_splits")
MASKS_ROOT  = Path("pseudo_masks")

# Class folders in your dataset
CLASSES = ["fire", "nofire"]

# We’ll accept both jpg and png
IMG_EXTS = (".jpg", ".jpeg", ".png")
