# Building-Level Earthquake Damage Assessment using CNN

This repository contains the implementation of a CNN-based framework for
earthquake-induced damage assessment at the building level.

## Input Features
Each building is represented by a numerical feature vector including:
- Pre-seismic SAR backscatter
- Post-seismic SAR backscatter
- Pre-seismic SAR coherence
- co-seismic SAR coherence
- Geological class
- Peak Ground Acceleration (PGA)
- Vs30 (shear-wave velocity)

## Model Description
- Architecture: 1D Convolutional Neural Network (Conv1D)
- Input: 7 numerical features per building
- Output: Binary damage classification (damaged / undamaged)
- Epochs: 50
- Batch size: 150
- Validation split: 50%

## Usage
```bash
pip install -r requirements.txt
python src/model_train_with_histograms.py

