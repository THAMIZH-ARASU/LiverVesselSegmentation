#!/usr/bin/env python3
"""
Test script for LiVS handler - validates the handler works correctly with the actual dataset
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing.livs import LiVSHandler

def test_livs_handler():
    """Test the LiVS handler with the actual dataset"""
    
    # Dataset path
    dataset_path = "/run/media/Thamizh/THAMIZH/LiVS"
    
    print("=" * 60)
    print("Testing LiVS Handler")
    print("=" * 60)
    print(f"\nDataset path: {dataset_path}\n")
    
    # Initialize handler
    print("1. Initializing LiVSHandler...")
    handler = LiVSHandler(dataset_path)
    print(f"   ✓ Handler initialized")
    print(f"   - Image directory: {handler.volumes_dir}")
    print(f"   - Vessel mask directory: {handler.vessel_masks_dir}")
    
    # Validate dataset
    print("\n2. Validating dataset structure...")
    is_valid = handler.validate_dataset()
    if is_valid:
        print("   ✓ Dataset structure is valid")
    else:
        print("   ✗ Dataset structure validation failed")
        return False
    
    # Get subject list
    print("\n3. Getting subject list...")
    subjects = handler.get_subject_list()
    print(f"   ✓ Found {len(subjects)} subjects")
    
    if len(subjects) == 0:
        print("   ✗ No subjects found!")
        return False
    
    # Display first 5 subjects
    print("\n4. First 5 subjects:")
    for i, subject in enumerate(subjects[:5]):
        print(f"   Subject {i+1}:")
        print(f"     - ID: {subject['subject_id']}")
        print(f"     - Image: {Path(subject['image']).name}")
        print(f"     - Vessel mask: {Path(subject['vessel_label']).name}")
    
    # Validate a few vessel masks
    print("\n5. Validating vessel masks (first 3 subjects)...")
    for i, subject in enumerate(subjects[:3]):
        is_valid = handler.validate_vessel_labels(subject['vessel_label'])
        status = "✓" if is_valid else "✗"
        print(f"   {status} {subject['subject_id']}: {'Valid' if is_valid else 'Invalid'}")
    
    # Check label statistics
    print("\n6. Checking label statistics (first subject)...")
    try:
        import nibabel as nib
        import numpy as np
        
        first_subject = subjects[0]
        mask_img = nib.load(first_subject['vessel_label'])
        mask_data = mask_img.get_fdata()
        
        unique_labels = np.unique(mask_data)
        vessel_voxels = np.sum(mask_data == 1)
        total_voxels = mask_data.size
        vessel_percentage = (vessel_voxels / total_voxels) * 100
        
        print(f"   Subject: {first_subject['subject_id']}")
        print(f"   - Shape: {mask_data.shape}")
        print(f"   - Unique labels: {unique_labels}")
        print(f"   - Vessel voxels: {vessel_voxels:,} ({vessel_percentage:.2f}%)")
        print(f"   - Total voxels: {total_voxels:,}")
        
    except Exception as e:
        print(f"   ✗ Error checking statistics: {e}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_livs_handler()
    sys.exit(0 if success else 1)
