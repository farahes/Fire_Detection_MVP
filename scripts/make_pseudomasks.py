import cv2, torch, numpy as np, torchvision as tv
from pathlib import Path
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.config import SPLITS_ROOT, MASKS_ROOT

sz=224
device='cuda' if torch.cuda.is_available() else 'cpu'

# load classifier + get index for "fire"
clf=tv.models.mobilenet_v3_small(weights=None)
clf.classifier[3]=torch.nn.Linear(clf.classifier[3].in_features, 2)
clf.load_state_dict(torch.load("cls_best.pth", map_location=device))
clf.eval().to(device)

# map class name -> idx using training folder (safe since ImageFolder orders alphabetically)
train_ds=tv.datasets.ImageFolder(SPLITS_ROOT/'train', transform=T.ToTensor())
class_to_idx=train_ds.class_to_idx  # e.g., {'fire':0,'nofire':1} or vice versa
fire_idx=class_to_idx['fire']

# target Grad-CAM layer
target_layers=[clf.features[-1]]
cam=GradCAM(model=clf, target_layers=target_layers)

def save_mask(img_path: Path, out_dir: Path, is_fire: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not is_fire:
        # blank mask for nofire
        img=cv2.imread(str(img_path))
        h,w=img.shape[:2]
        m=np.zeros((h,w), np.uint8)
        cv2.imwrite(str(out_dir/f"{img_path.stem}.png"), m)
        return
    # CAM for fire class
    img=cv2.imread(str(img_path))[:,:,::-1]
    img_res=cv2.resize(img,(sz,sz)).astype(np.float32)/255.0
    tensor=T.ToTensor()(img_res).unsqueeze(0).to(device)
    grayscale_cam=cam(input_tensor=tensor, targets=[ClassifierOutputTarget(fire_idx)])[0]
    # threshold + upsize
    m=(grayscale_cam>0.40).astype(np.uint8)*255
    m=cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    m=cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir/f"{img_path.stem}.png"), m)

def process_split(split):
    for cls in ["fire","nofire"]:
        img_dir=SPLITS_ROOT/split/cls
        out_dir=MASKS_ROOT/split/cls
        is_fire=(cls=="fire")
        for p in img_dir.glob("*"):
            if p.suffix.lower() not in [".jpg",".jpeg",".png"]: 
                continue
            save_mask(p, out_dir, is_fire)

for split in ["train","val","test"]:
    process_split(split)
print("Pseudo-masks saved under", MASKS_ROOT)
