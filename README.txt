# Fire-MVP (Tiny Segmentation via Weak Supervision)
Laptop MVP: fire/no-fire classifier → Grad-CAM pseudo-masks → tiny SegFormer-B0 segmentation.

Note: install all imports pip instal... to ensure all files work.

Step by step to run this on you  laptop: 

Re-train the classifier: python -m scripts.train_cls
Launch the webcam detector: python scripts/webcam_fire_cls.py

Point it at a person → should say NOFIRE. 
Point it at a real flame or a flame video → should flip to FIRE, with a heatmap
