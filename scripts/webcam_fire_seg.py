import cv2, torch, numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = SegformerFeatureExtractor(align=False, reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load("seg_best.pth", map_location=DEVICE))
model.eval().to(DEVICE)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Cannot open camera.")

print("Press 'q' to quit.")
with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc = feature_extractor(images=[rgb], return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        pred = up.argmax(1)[0].cpu().numpy()  # 0=background, 1=fire

        overlay = frame.copy()
        overlay[pred==1] = [0,0,255]  # red on fire pixels
        vis = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        cv2.imshow("Fire Segmentation (SegFormer-B0)", vis)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
