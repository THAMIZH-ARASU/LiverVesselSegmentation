# LiVS Dataset Handler - Implementation Guide

## Overview
The LiVS (Liver Vessel Segmentation) dataset handler has been successfully implemented and tested with the actual dataset structure.

## Dataset Structure
```
/run/media/Thamizh/THAMIZH/LiVS/
├── image_nii/           # CT scan images
│   ├── case0001.nii.gz
│   ├── case0002.nii.gz
│   └── ... (532 cases total)
└── vessel_mask_nii/     # Vessel segmentation masks
    ├── case0001.nii.gz
    ├── case0002.nii.gz
    └── ... (532 cases total)
```

## Implementation Details

### 1. Handler File: `data_preprocessing/livs.py`
- **Class**: `LiVSHandler`
- **Dataset Type**: `"livs"`
- **Directories**:
  - Images: `image_nii/`
  - Vessel Masks: `vessel_mask_nii/`
- **File Pattern**: `case*.nii.gz`
- **Label Format**: Binary (0: background, 1: vessel)

### 2. Updated Files

#### `data_preprocessing/data_preprocessor.py`
Added LiVS support to the dataset handler factory:
```python
def get_dataset_handler(self, dataset_path: str, dataset_type: str) -> DatasetHandler:
    if dataset_type == "hepatic_vessel":
        return HepaticVesselHandler(dataset_path)
    elif dataset_type == "dircadb_vessel":
        return DircadbVesselHandler(dataset_path)
    elif dataset_type == "livs":
        return LiVSHandler(dataset_path)  # NEW
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
```

#### `run_data_preprocessing.py`
Added `"livs"` to the dataset type choices:
```python
parser.add_argument("--dataset_type", type=str, required=True,
                   choices=["hepatic_vessel", "dircadb_vessel", "livs"],
                   help="Type of dataset (hepatic_vessel for MSD; dircadb_vessel for 3DIRCADB; livs for LiVS)")
```

## Testing Results

### Test Script: `test_livs_handler.py`
Successfully validated:
- ✓ Handler initialization
- ✓ Dataset structure validation
- ✓ Subject list extraction (532 subjects found)
- ✓ Vessel mask validation
- ✓ Label statistics

### Sample Output:
```
============================================================
Testing LiVS Handler
============================================================

Dataset path: /run/media/Thamizh/THAMIZH/LiVS

1. Initializing LiVSHandler...
   ✓ Handler initialized
   - Image directory: /run/media/Thamizh/THAMIZH/LiVS/image_nii
   - Vessel mask directory: /run/media/Thamizh/THAMIZH/LiVS/vessel_mask_nii

2. Validating dataset structure...
   ✓ Dataset structure is valid

3. Getting subject list...
   ✓ Found 532 subjects

4. First 5 subjects:
   Subject 1:
     - ID: livs_case0001
     - Image: case0001.nii.gz
     - Vessel mask: case0001.nii.gz
   ...

5. Validating vessel masks (first 3 subjects)...
   ✓ livs_case0001: Valid
   ✓ livs_case0002: Valid
   ✓ livs_case0003: Valid

6. Checking label statistics (first subject)...
   Subject: livs_case0001
   - Shape: (256, 256, 39)
   - Unique labels: [0. 1.]
   - Vessel voxels: 6,458 (0.25%)
   - Total voxels: 2,555,904

============================================================
✓ All tests passed!
============================================================
```

## Usage

### Command Line Usage

To preprocess the LiVS dataset, use:

```bash
# Activate virtual environment
source venv/bin/activate

# Install required dependencies (if not already installed)
pip install torch pytorch-lightning torchio nibabel numpy pandas scipy scikit-learn scikit-image matplotlib tqdm

# Run preprocessing
python run_data_preprocessing.py \
    --dataset_path /run/media/Thamizh/THAMIZH/LiVS \
    --dataset_type livs \
    --output_dir ./data_preprocessed/livs

# With custom configuration
python run_data_preprocessing.py \
    --dataset_path /run/media/Thamizh/THAMIZH/LiVS \
    --dataset_type livs \
    --output_dir ./data_preprocessed/livs \
    --target_spacing 1.0 1.0 1.0 \
    --target_size 256 256 256 \
    --intensity_range -100 400 \
    --normalize_method zscore
```

### Python API Usage

```python
from data_preprocessing.livs import LiVSHandler

# Initialize handler
handler = LiVSHandler("/run/media/Thamizh/THAMIZH/LiVS")

# Validate dataset
if handler.validate_dataset():
    print("Dataset is valid!")
    
# Get subject list
subjects = handler.get_subject_list()
print(f"Found {len(subjects)} subjects")

# Access first subject
first_subject = subjects[0]
print(f"ID: {first_subject['subject_id']}")
print(f"Image: {first_subject['image']}")
print(f"Vessel Label: {first_subject['vessel_label']}")
```

## Key Features

1. **Automatic File Matching**: Automatically pairs image and vessel mask files by case number
2. **Validation**: Validates dataset structure and label consistency
3. **Subject Metadata**: Returns structured metadata for each subject with proper paths
4. **Integration**: Seamlessly integrates with existing preprocessing pipeline
5. **Error Handling**: Warns about missing or invalid files

## Dependencies

### Minimal (for testing handler only):
```bash
pip install nibabel numpy pydicom
```

### Full (for preprocessing):
```bash
pip install torch pytorch-lightning torchio nibabel numpy scipy scikit-learn tqdm
```

## Next Steps

To run the full preprocessing pipeline:

1. **Install all dependencies** (see above)
2. **Run the preprocessing** command with desired configuration
3. **Output** will be saved to the specified output directory with:
   - Preprocessed images
   - Preprocessed labels
   - Metadata JSON files
   - Train/val/test splits

## Notes

- The handler expects binary vessel masks (0: background, 1: vessel)
- All 532 cases are detected and validated
- File naming follows the pattern: `case####.nii.gz`
- Compatible with the existing vessel segmentation preprocessing pipeline
