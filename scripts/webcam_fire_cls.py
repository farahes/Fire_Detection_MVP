# scripts/webcam_fire_cls.py
import cv2, torch, torchvision as tv, torch.nn as nn
import numpy as np, collections, time
from torchvision import transforms as T

# --------- settings
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES  = ["fire", "nofire"]        # keep consistent with training
IMG_SIZE = 224
SHOW_CAM = True                      # Grad-CAM overlay toggle (we'll gate it)
THRESH_CLS  = 0.60                   # classifier prob threshold for "fire"
THRESH_AREA = 0.015                  # % of frame that must be fire-colored
SMOOTH_N    = 5                      # temporal smoothing window

# --------- model
model = tv.models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("cls_best.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# (optional) Grad-CAM
try:
    from pytorch_grad_cam import EigenCAM   # no gradients required; simpler & fast
    cam = EigenCAM(model=model, target_layers=[model.features[-1]])
    HAVE_CAM = True
except Exception as e:
    print("CAM overlay disabled:", e)
    HAVE_CAM = False

tfm = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

def draw_label(frame, text, color=(0,255,0)):
    cv2.rectangle(frame, (10,10), (520,58), (0,0,0), -1)
    cv2.putText(frame, text, (18,48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

def overlay_heatmap(frame_bgr, heatmap):
    heatmap = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return cv2.addWeighted(heatmap, 0.35, frame_bgr, 0.65, 0)

def fire_color_mask(bgr):
    """Simple HSV fire prior: red/orange/yellow with decent saturation & brightness."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # two hue bands to catch reds wrapping around 180, and orange/yellow
    lower1 = np.array([0,   120, 150], dtype=np.uint8)   # red
    upper1 = np.array([15,  255, 255], dtype=np.uint8)

    lower2 = np.array([16,  100, 160], dtype=np.uint8)   # orange/yellow
    upper2 = np.array([50,  255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # clean noise
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)
    return mask

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try cv2.VideoCapture(0) if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
if not cap.isOpened():
    raise SystemExit("Enable camera access in Windows Privacy & security → Camera.")

print("q: quit | g: toggle heatmap")
probs_q = collections.deque(maxlen=SMOOTH_N)
areas_q = collections.deque(maxlen=SMOOTH_N)

while True:
    ok, frame = cap.read()
    if not ok: break

    # classifier
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = tfm(tv.transforms.functional.to_pil_image(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    p_fire = float(prob[CLASSES.index("fire")])
    probs_q.append(p_fire)

    # color prior
    mask = fire_color_mask(frame)
    area = (mask > 0).mean()         # fraction of pixels flagged as "fire color"
    areas_q.append(area)

    # smooth
    p_fire_s = sum(probs_q) / len(probs_q)
    area_s   = sum(areas_q) / len(areas_q)

    # fuse (AND): require both confidence AND enough fire-colored pixels
    is_fire = (p_fire_s >= THRESH_CLS) and (area_s >= THRESH_AREA)

    # draw
    out = frame.copy()
    txt = f"{'FIRE' if is_fire else 'NOFIRE'}  {p_fire_s*100:.1f}%  | color:{area_s*100:.1f}%"
    color = (0,0,255) if is_fire else (0,255,0)
    draw_label(out, txt, color=color)

    # overlay heatmap only if is_fire and CAM available
    if SHOW_CAM and HAVE_CAM and is_fire:
        # use predicted class for stability
        grayscale_cam = cam(input_tensor=x)[0]   # EigenCAM ignores targets
        out = overlay_heatmap(out, grayscale_cam)

    # optional: show color mask thumbnail for debugging
    small_mask = cv2.resize(mask, (160, 90))
    out[10:10+90, out.shape[1]-10-160:out.shape[1]-10] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Fire / NoFire (Classifier + Color Prior)", out)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('g'): SHOW_CAM = not SHOW_CAM

cap.release()
cv2.destroyAllWindows()
