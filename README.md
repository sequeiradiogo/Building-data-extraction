# Building Extraction from Satellite Images

This repository implements a semantic segmentation pipeline to detect buildings from satellite imagery. The project balances computational efficiency and model performance using a U-Net architecture.

## Repository Structure

- `prepare_dataset.py`: Prepares the dataset for PyTorch.
- `dataset.py`: Converts images and masks into tensors.
- `trainv1.py`: Initial training attempt with a simple U-Net.
- `train_final_version.py`: Final training script using ResNet-34 U-Net.
- `test.py`: Runs inference and generates submission results.

## Model Choice

The chosen model is U-Net, suitable for pixel-wise classification (building vs. background). U-Net combines accuracy with computational efficiency.

## Architecture

- **Initial model**: 3 downsampling blocks, 2 upsampling blocks. Metrics were low (IoU < 0.4, F1 < 0.5), with blurry predicted masks.
- **Final model**: ResNet-34 encoder, 4 decoder blocks. Mixed precision (torch.amp.autocast + GradScaler) reduced memory usage. Optimizer: AdamW. Loss emphasized F1-score, with IoU, precision, and recall monitored. Achieved IoU: 0.7833.

## Metrics

- **Main metrics**: IoU, BCE loss, F1-score.
- BCE encourages pixel-wise accuracy; F1 and IoU assess mask alignment.
- Visual inspection of predicted masks was performed on batches.

## Inference

- Loads trained weights, resizes images, and generates binary masks.
- Small noise in masks is removed.
- Outputs: visual masks and CSV file for Kaggle submission.

## Results & Observations

- The final model outperformed the first, mainly due to higher CNN complexity and emphasis on F1-score.
- Longer training with learning rate adjustments could improve results.
- Kaggle submission score: 0.26. Likely cause: mask-to-polygon conversion in `test.py`.

