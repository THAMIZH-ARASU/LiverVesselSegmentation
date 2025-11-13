# Hepatic Vessel Segmentation Data Preprocessing Module

This module provides tools and pipelines for preprocessing CT hepatic vessel datasets for segmentation tasks, adapted for vessel-only tasks.

## Main Components
- **data_preprocessor.py**: Orchestrates the preprocessing workflow, including dataset splitting, transformation, and saving outputs.
- **handler.py**: Base class for dataset handlers.
- **msd_handler.py** (renamed: HepaticVesselHandler): Handler for the MSD Hepatic Vessel dataset (NIfTI format).
- **intensity_normalizer.py**: Implements intensity normalization methods (z-score, min-max, robust).
- **subject_metadata.py**: Dataclass for storing subject metadata.
- **pipelines/data_preprocessing_pipeline.py**: Main pipeline for spatial and intensity preprocessing, augmentation, and metadata extraction.

## Features
- **Supports MSD Hepatic Vessel dataset** (.nii/.nii.gz, label==1 for vessel)
- **Spatial preprocessing**: Resampling, resizing
- **Intensity preprocessing**: Clipping, normalization
- **Data augmentation** (optional, for training)
- **Vessel-only mask output**: Only label==1 used in mask; all others set to 0
- **Metadata extraction**: Spacing, shape, origin, direction, affine, intensity stats
- **Train/val/test split**
- **Saves preprocessed images, labels, and metadata**

## Usage

### Command Line
Run the preprocessing pipeline with:
```bash
python3 run_data_preprocessing.py \
  --dataset_path /path/to/dataset \
  --dataset_type hepatic_vessel \
  --output_dir data_preprocessed \
  [--config /path/to/config.json]
```

### Example
```bash
python3 run_data_preprocessing.py \
  --dataset_path /home/user/Data/MSD_Hepatic \
  --dataset_type hepatic_vessel \
  --output_dir data_preprocessed
```

### Configuration
Specify preprocessing parameters via CLI or JSON config. Key parameters include:
- `target_spacing`: Target voxel spacing (default: [1.0, 1.0, 1.0])
- `target_size`: Target image size (default: [256, 256, 256])
- `intensity_range`: Intensity clipping range (default: [-100, 400])
- `normalize_method`: Normalization method (`zscore`, `minmax`, `robust`)
- `apply_augmentation`: Enable/disable augmentation (default: True)
- `train_ratio`, `val_ratio`, `test_ratio`: Dataset split ratios

## Input Structure
- `imagesTr/`: CT images (`.nii` or `.nii.gz`)
- `labelsTr/`: Vessel masks (`.nii` or `.nii.gz`), only label==1 is considered vessel

## Output Structure
- `data_preprocessed/`
  - `train/`, `val/`, `test/`: Preprocessed splits
    - `{subject_id}_image.nii.gz`: Preprocessed image
    - `{subject_id}_label.nii.gz`: Preprocessed vessel mask (only label==1 retained)
  - `preprocessing_config.json`: Used configuration
  - `preprocessing_metadata.pkl`: Metadata for all subjects
  - `preprocessing_summary.txt`: Human-readable summary

## Note
- The pipeline is adapted for vessel segmentation only. All references to liver/tumor have been removed.
- Vessel handler requires the vessel label to be present (label==1) in each mask.

