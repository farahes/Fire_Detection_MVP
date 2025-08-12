import cv2, torch, numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from src.config import SPLITS_ROOT

model=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2)
model.load_state_dict(torch.load("seg_best.pth", map_location="cpu")); model.eval()
extractor=SegformerFeatureExtractor(align=False, reduce_labels=False)

def run(img_path: str):
    img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    enc=extractor(images=[img], return_tensors="pt")
    with torch.no_grad():
        out=model(**enc).logits
    up=torch.nn.functional.interpolate(out, size=img.shape[:2], mode="bilinear", align_corners=False)
    pred=up.argmax(1)[0].cpu().numpy()
    overlay=img.copy(); overlay[pred==1]=[255,0,0]
    vis=cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.imwrite("preview.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("Saved preview.png")

# Example: pick any file from data_splits/test/fire/
# run("data_splits/test/fire/xxx.jpg")
