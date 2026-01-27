# RSEdit-Light

This directory contains the core implementation of **RSEdit**, a unified framework for instruction-following remote sensing image editing.

## Overview

RSEdit adapts pretrained text-to-image diffusion models (U-Net and DiT) into specialized editors for Earth observation data. By leveraging channel concatenation and in-context token concatenation, it preserves geospatial content while performing precise, physically coherent edits based on natural language instructions.

For more details, visit our [Project Page](https://d-sakura.github.io/RSEdit-preview/).

## Key Features

- **Unified Framework**: Supports both U-Net (e.g., SD 1.5/2.1) and DiT (e.g., SD3/Flux) backbones.
- **RS-Specific Conditioning**: Tailored alignment for bi-temporal structure and spatial priors.
- **Robust Generalization**: Effective across disaster impacts, urban growth, and seasonal shifts.

## Training

The training scripts for different backbones are provided:
- `train_unet.py`: Training script for U-Net based RSEdit.
- `train_dit.py`: Training script for DiT based RSEdit.
- `train_sd3.py`: Training script for SD3 based RSEdit.

Refer to `train.sh` for example training commands.

## Inference Pipelines

- `pipeline_rsedit_dit.py`: Inference pipeline for DiT-based models.
- `pipeline_rsedit_sd3.py`: Inference pipeline for SD3-based models.
