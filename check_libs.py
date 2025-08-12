import sys, torch, cv2, transformers, pytorch_grad_cam
from pytorch_grad_cam import GradCAM

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cv2:", cv2.__version__)
print("transformers:", transformers.__version__)
print("grad-cam:", pytorch_grad_cam.__version__)
