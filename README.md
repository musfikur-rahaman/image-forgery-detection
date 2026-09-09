# Image Forgery Detection: ResNet50 + U-Net + Grad-CAM

An explainable deep-learning framework for image forgery detection and localization — combining **classification**, **pixel-level segmentation**, and **visual interpretability** in one unified pipeline for trustworthy forensic analysis.

![System Overview](assets/system.png)

## The pipeline

1. **ResNet50 classifier** — every input image is classified as real or forged.
2. **U-Net segmentation** — images flagged as forged go through pixel-level localization of the manipulated region.
3. **Grad-CAM explainability** — heatmaps show which regions drove the classifier's decision, so the verdict is auditable, not a black box.

## Results (CASIA 2.0)

| Task | Metric | Score |
|---|---|---|
| Classification | Accuracy | 80.32% |
| Classification | AUC | 0.86 |
| Classification | Forged recall | 0.88 |
| Classification | F1 | 0.80 |
| Segmentation | Dice | ~0.21 |
| Segmentation | IoU | ~0.15 |

High forged-image recall (0.88) is the metric that matters most here — in forensics, a missed forgery is worse than a false alarm. Segmentation scores are modest, consistent with CASIA 2.0's extremely small manipulated regions and noisy ground-truth masks. Grad-CAM confirms the classifier keys on texture inconsistencies, boundary artifacts, and splicing regions — genuine forensic cues.

## Methods

**Classifier** — ResNet50 pretrained on ImageNet, original FC layer replaced with a custom head (2048→256, batch norm, ReLU, dropout, 256→1). Trained with `BCEWithLogitsLoss` and class weighting (`pos_weight`) for the real/forged imbalance; augmentation includes rotation, cropping, color jitter, and blur. Decision threshold tuned on the validation set.

**Segmentation** — U-Net trained from scratch on forged images with ground-truth masks (filename-normalized image–mask pairing), optimized with combined BCE + Dice loss to handle severe foreground–background imbalance.

**Explainability** — Grad-CAM applied to the trained classifier at inference time.

## Dataset

[CASIA 2.0 Image Tampering Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset) — 12,000+ authentic and tampered images (copy–move and splicing) with binary ground-truth masks for a subset. Classification split 70/15/15 stratified (train: 5,243 real / 4,002 forged); segmentation pairs split 80/20.

## Repository contents

- `Image_Classification_Segmentation_CASIA2_ResNet50.ipynb` — the full pipeline: data loading, ResNet50 classifier, threshold tuning, U-Net segmentation, Grad-CAM visualization
- `MultiBranch_EfficientNetB3_DCT.ipynb` — a second approach: multi-branch EfficientNet-B3 (RGB) fused with an MLP on 64-bin DCT coefficient histograms, with Grad-CAM
- `assets/system.png` — system overview diagram
- `docs/Image_Forgery_Detection.pdf` — project report

## Quickstart

Both notebooks are designed for Google Colab with a GPU runtime. They expect the CASIA 2.0 dataset on Google Drive (see `DATA_ROOT` in cell 2 of each notebook and adjust the path).

```bash
pip install torch torchvision opencv-python matplotlib tqdm scikit-learn
```

## Limitations & future work

- U-Net localization is constrained by noisy CASIA 2.0 masks and tiny forged regions; a pretrained encoder should help.
- Occasional false positives on strong shadows, lighting variation, and complex textures.
- CASIA covers copy–move/splicing only — generalization to deepfakes and AI-generated images needs broader training data.
- Next step: real-time forensic verification tool.

## References

- He et al. Deep Residual Learning for Image Recognition. *CVPR*, 2016. (ResNet50)
- Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*, 2015.
- Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*, 2017.
- Dong et al. CASIA Image Tampering Detection Evaluation Database. *IEEE ChinaSIP*, 2013.

## Author

Musfikur Rahaman — PhD student, UA Little Rock
