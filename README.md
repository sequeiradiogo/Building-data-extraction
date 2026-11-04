# Building Extraction from Aerial Imagery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sequeiradiogo/Building-data-extraction/blob/main/notebooks/inference_demo.ipynb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Detecting building footprints in aerial images and exporting them as polygons, built for the
[IEEE BigData Cup 2024 – Building Extraction Generalization Challenge](https://www.kaggle.com/competitions/building-extraction-generalization-2024) on Kaggle.

The pipeline treats the task as **semantic segmentation**: a U-Net with an ImageNet-pretrained
ResNet-34 encoder predicts a building mask, which is cleaned up and converted into one simplified
polygon per connected region.

![Predictions on competition test images](assets/predictions.jpg)
<sub>Predictions on the 9 sample images from the competition test set in `output/examples/test_images.zip`
(no ground truth is available for them). Orange: predicted mask, yellow: exported polygons. Note the
false positive along the right edge of `0046.tif`.</sub>

---

## Pipeline

```
COCO polygons ──► prepare_dataset.py ──► binary masks (PNG)
                                              │
                                              ▼
                      train.py: U-Net / ResNet-34, BCE + Dice loss, AdamW + OneCycleLR, AMP
                                              │
                                              ▼
test images ──► predict.py: mask ► morphological clean-up ► contours ► simplified polygons ──► submission.csv
```

| Component | Choice |
|---|---|
| Architecture | U-Net, ResNet-34 encoder (ImageNet weights), 24.4 M parameters ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)) |
| Input | 512 × 512, ImageNet normalization |
| Augmentation | flips, 90° rotations, small rotations, brightness/contrast, blur/noise ([Albumentations](https://albumentations.ai)) |
| Loss | 0.3 · BCE + 0.7 · Dice (Dice weighted higher to favour overlap with the building mask) |
| Optimization | AdamW, OneCycleLR (peak 1e-3), mixed precision, early stopping on validation IoU |
| Post-processing | threshold 0.5, morphological close + open, contours with area ≥ 100 px, Douglas–Peucker simplification (ε = 1 % of the perimeter) |

A simpler **baseline** (`train_baseline.py`) is included for comparison: a 3-level U-Net trained from
scratch with an equally weighted BCE + Dice loss. The final model performed clearly better.

## Results

| | Validation IoU | Validation Dice / F1 | Kaggle score |
|---|---|---|---|
| U-Net / ResNet-34 | **0.783** | **0.873** | ~0.26 |

Validation metrics are pixel-level, averaged per image.

**Why the leaderboard score is much lower than the validation IoU.** The competition scores the
submitted *polygons* (an F1 score over buildings), while this model is trained and validated on
*pixels*. Semantic segmentation cannot tell touching buildings apart, so a row of houses becomes a
single polygon (visible in the Chicago example above). Small buildings removed by the area filter and
the polygon simplification also cost matches. Pixel IoU does not penalize any of this, but a
per-building metric does. See [Next steps](#limitations-and-next-steps).

### Generalization to other imagery

To check how the model behaves outside the competition data, it was evaluated, without any
fine-tuning, on the 967 test images of the
[Roboflow *Buildings Instance Segmentation* dataset](https://huggingface.co/datasets/keremberke/satellite-building-segmentation) (CC BY 4.0):

| Imagery | Images | IoU | F1 |
|---|---|---|---|
| Aerial (Tyrol, Austin, Chicago) | 400 | 0.68 | 0.77 |
| Other sources, mostly satellite (Kherson, Tripoli, Donetsk, Mekele, Mykolaiv, Kharkiv, …) | 567 | 0.23 | 0.31 |
| **All** | 967 | 0.42 | 0.50 |

![Predictions on images from a different dataset](assets/generalization.jpg)
<sub>One randomly picked image per region. Orange: predicted mask, yellow: exported polygons, blue: ground truth.</sub>

The model transfers reasonably well to other **aerial** imagery but degrades strongly on **satellite**
images, which look quite different from the training data (see the Kharkiv example). This is the
same domain-shift problem that the "generalization" theme of the challenge is about.

## Repository structure

```
├── src/
│   ├── prepare_dataset.py   # COCO polygons -> image / binary mask pairs
│   ├── dataset.py           # PyTorch datasets + joint image/mask augmentation
│   ├── train.py             # final model (U-Net / ResNet-34)
│   ├── train_baseline.py    # baseline U-Net from scratch
│   ├── predict.py           # inference, mask -> polygons, submission CSV
│   ├── engine.py            # losses, metrics, train / validation loops
│   └── common.py            # checkpoints, model construction, figures
├── notebooks/
│   └── inference_demo.ipynb # inference demo (runs on CPU)
├── data/instructions.md     # where to get the dataset
├── assets/                  # README figures and sample images from another dataset
├── models/                  # model weights (downloaded, not tracked)
├── output/
│   ├── submission.csv       # submitted predictions
│   └── examples/            # example image/mask and 9 competition test images
└── requirements.txt
```

## Getting started

### Quick demo
Open [`notebooks/inference_demo.ipynb`](notebooks/inference_demo.ipynb) in Colab with the badge at the
top. It downloads the trained weights and runs on sample competition images; no dataset is needed.

### Installation
```bash
git clone https://github.com/sequeiradiogo/Building-data-extraction.git
cd Building-data-extraction
pip install -r requirements.txt
```

### Pre-trained weights
Download `unet_resnet34_buildings.pth` (98 MB) from the
[latest release](https://github.com/sequeiradiogo/Building-data-extraction/releases/latest) into `models/`:

```bash
mkdir -p models
curl -L -o models/unet_resnet34_buildings.pth \
  https://github.com/sequeiradiogo/Building-data-extraction/releases/download/v1.0/unet_resnet34_buildings.pth
```

SHA-256: `543215f09324e38a52be0c735a931c46ec415ea5836eda064f3bc8767d725093`

### Data
Download the data from the [competition page](https://www.kaggle.com/competitions/building-extraction-generalization-2024/data)
(see [`data/instructions.md`](data/instructions.md)) and arrange it as:

```
raw/
├── train/  train.json  image/*.jpg
├── val/    val.json    image/*.jpg
└── test/   *.tif   (or image/*.tif)
```

Then convert the COCO annotations into masks:

```bash
python src/prepare_dataset.py --raw-dir raw --out-dir data
```

This writes `data/{train,val}/{images,masks}`, `data/test/images` and a few overlay previews in
`data/previews/` to check that the masks line up with the images.

| Training image | Generated mask |
|---|---|
| <img src="output/examples/raw.jpg" width="300"> | <img src="output/examples/mask.jpg" width="300"> |

### Training
```bash
python src/train.py --data-dir data --checkpoint-dir models
python src/train_baseline.py --data-dir data --checkpoint-dir models   # optional baseline
```

A checkpoint is written after every epoch (`unet_resnet34_epoch{N}_IoU{x}.pth`), and training
**resumes automatically** from the latest one. This includes the optimizer, LR schedule, AMP scaler and
early-stopping state, which matters on Colab where sessions disconnect. Run `--help` to see all
hyperparameters, and use `--no-resume` to start over.

### Inference
```bash
python src/predict.py --test-dir data/test/images \
                      --checkpoint models/unet_resnet34_buildings.pth \
                      --output-csv output/submission.csv
```

Without `--checkpoint`, the checkpoint with the best validation IoU in `--checkpoint-dir` is used.
Use `--mask-dir` to also save the predicted masks; `--threshold`, `--min-area` and `--simplify-eps`
control the post-processing. The CSV has one row per image: `ImageID`, followed by a list of polygons
`[[(x,y),(x,y),...],[...]]` in pixel coordinates.

### Running on Google Colab with Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/sequeiradiogo/Building-data-extraction.git
%cd Building-data-extraction
!pip install -q -r requirements.txt
!python src/train.py --data-dir /content/drive/MyDrive/data --checkpoint-dir /content/drive/MyDrive/models
```

Keeping the checkpoints on Drive lets training continue in a new session after a disconnect.

## Limitations and next steps

- **Separate touching buildings.** This is the most important improvement for the leaderboard metric.
  Options: a third "building boundary" class, watershed on the distance transform of the mask, or an
  instance-segmentation model (Mask R-CNN / YOLO-seg). The winning solution of the challenge used YOLOv8-seg.
- **Tune the post-processing** (`--threshold`, `--min-area`, `--simplify-eps`) against the competition
  metric instead of pixel IoU.
- **Longer training and test-time augmentation** (flips) for more accurate masks.
- **More diverse training data**, for example satellite imagery, to generalize beyond aerial photos.

## Acknowledgements

- Competition data: IEEE BigData Cup 2024 – Building Extraction Generalization Challenge.
- The images in `assets/samples/` and `assets/generalization.jpg` come from the
  [Roboflow Buildings Instance Segmentation dataset](https://universe.roboflow.com/roboflow-universe-projects/buildings-instance-segmentation),
  distributed on [Hugging Face](https://huggingface.co/datasets/keremberke/satellite-building-segmentation)
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Model implementation: [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

## License

Code released under the [MIT License](LICENSE).
