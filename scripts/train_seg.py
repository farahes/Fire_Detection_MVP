import torch, cv2, numpy as np, os
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from src.config import SPLITS_ROOT, MASKS_ROOT

size=384; bs=4; epochs=15; lr=5e-5
device='cuda' if torch.cuda.is_available() else 'cpu'

class FireSegDS(Dataset):
    def __init__(self, split):
        self.samples=[]
        for cls in ["fire","nofire"]:
            img_dir=SPLITS_ROOT/split/cls
            msk_dir=MASKS_ROOT/split/cls
            for f in os.listdir(img_dir):
                if not f.lower().endswith((".jpg",".jpeg",".png")): 
                    continue
                img=str(img_dir/f)
                msk=str(msk_dir/f.rsplit(".",1)[0]+".png")
                self.samples.append((img, msk))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img_path, msk_path = self.samples[i]
        img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        msk=cv2.imread(msk_path,0)
        img=cv2.resize(img,(size,size))
        msk=cv2.resize(msk,(size,size), interpolation=cv2.INTER_NEAREST)
        msk=(msk>0).astype(np.int64)  # 1=fire
        return img, msk

extractor=SegformerFeatureExtractor(align=False, reduce_labels=False)
def collate(batch):
    imgs, msks = zip(*batch)
    enc = extractor(list(imgs), masks=list(msks), return_tensors="pt")
    return enc

train_dl=DataLoader(FireSegDS("train"), batch_size=bs, shuffle=True,  collate_fn=collate, num_workers=4)
val_dl  =DataLoader(FireSegDS("val"),   batch_size=bs, shuffle=False, collate_fn=collate, num_workers=4)

model=SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True
).to(device)

optim=torch.optim.AdamW(model.parameters(), lr=lr)
best=0.0
for ep in range(epochs):
    model.train()
    for batch in train_dl:
        for k in batch: batch[k]=batch[k].to(device)
        optim.zero_grad(); out=model(**batch); out.loss.backward(); optim.step()

    # simple IoU on 'fire'
    model.eval(); inter=union=0
    with torch.no_grad():
        for batch in val_dl:
            for k in batch: batch[k]=batch[k].to(device)
            logits=model(**batch).logits
            up=torch.nn.functional.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)
            pred=up.argmax(1); gt=batch["labels"]
            i=((pred==1)&(gt==1)).sum().item()
            u=((pred==1)|(gt==1)).sum().item()
            inter+=i; union+=u
    iou=(inter/union) if union>0 else 0.0
    print(f"epoch {ep+1}: val IoU_fire={iou:.3f}")
    if iou>best:
        best=iou
        torch.save(model.state_dict(),"seg_best.pth")
print("best IoU_fire:", best)
