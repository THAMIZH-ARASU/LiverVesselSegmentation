"""
livs_handler.py

Handler for the LiVS (Liver Vessel Segmentation) dataset format. Extracts subject lists and validates dataset structure for preprocessing pipelines.

The LiVS dataset structure:
- image_nii/ (folder): Contains CT nifti files (case0001.nii.gz, case0002.nii.gz, etc.)
- vessel_mask_nii/ (folder): Contains vessel annotation nifti files (case0001.nii.gz, case0002.nii.gz, etc.)
"""
from typing import Dict, List
from pathlib import Path
import nibabel as nib
import numpy as np

from data_preprocessing.handler import DatasetHandler


class LiVSHandler(DatasetHandler):
    """
    Handler for LiVS (Liver Vessel Segmentation) dataset format.

    Attributes:
        volumes_dir (Path): Directory containing CT volume files.
        vessel_masks_dir (Path): Directory containing vessel segmentation annotation files.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the LiVSHandler.

        Args:
            dataset_path (str): Path to the dataset root.
        """
        super().__init__(dataset_path, "livs")
        self.volumes_dir = self.dataset_path / "image_nii"
        self.vessel_masks_dir = self.dataset_path / "vessel_mask_nii"
        
    def validate_dataset(self) -> bool:
        """
        Validate the LiVS dataset structure (image_nii and vessel_mask_nii directories).

        Returns:
            bool: True if required directories exist, False otherwise.
        """
        if not self.volumes_dir.exists():
            print(f"Error: Expected directory 'image_nii' not found in {self.dataset_path}")
            return False
            
        if not self.vessel_masks_dir.exists():
            print(f"Error: Expected directory 'vessel_mask_nii' not found in {self.dataset_path}")
            return False
            
        # Check if there are any volume files
        volume_files = list(self.volumes_dir.glob("case*.nii*"))
        if not volume_files:
            print(f"Error: No volume files found in {self.volumes_dir}")
            return False
            
        # Check if there are any vessel mask files
        vessel_mask_files = list(self.vessel_masks_dir.glob("case*.nii*"))
        if not vessel_mask_files:
            print(f"Error: No vessel mask files found in {self.vessel_masks_dir}")
            return False
            
        return True
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """
        Get subject list for LiVS format.

        Returns:
            List[Dict[str, str]]: List of subject metadata dicts with image and vessel_label paths.
        """
        subjects = []
        
        # Find all volume files (case0001.nii.gz, case0002.nii.gz, etc.)
        for volume_file in sorted(self.volumes_dir.glob("case*.nii*")):
            # Extract case number from filename (e.g., case0001.nii.gz -> 0001)
            case_name = volume_file.stem.replace(".nii", "")  # Remove .nii from .nii.gz
            vessel_mask_file = self.vessel_masks_dir / volume_file.name
            
            if vessel_mask_file.exists():
                subjects.append({
                    "subject_id": f"livs_{case_name}",
                    "image": str(volume_file),
                    "vessel_label": str(vessel_mask_file)
                })
            else:
                print(f"Warning: Vessel mask file not found for {volume_file.name}")
                
        return subjects
    
    def validate_vessel_labels(self, vessel_mask_path: str) -> bool:
        """
        Validate that the vessel mask file contains valid vessel labels.
        
        Args:
            vessel_mask_path (str): Path to the vessel mask file.
            
        Returns:
            bool: True if valid vessel labels found (contains label 1), False otherwise.
        """
        try:
            seg_img = nib.load(vessel_mask_path)
            seg_data = seg_img.get_fdata()
            
            # Check for valid vessel labels (0: background, 1: vessel)
            unique_labels = np.unique(seg_data)
            valid_labels = {0, 1}
            
            if not set(unique_labels).issubset(valid_labels):
                print(f"Warning: Unexpected labels found in {vessel_mask_path}: {unique_labels}")
                return False
                
            # Check if vessel (label 1) is present
            if 1 not in unique_labels:
                print(f"Warning: No vessel labels found in {vessel_mask_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating vessel mask {vessel_mask_path}: {str(e)}")
            return False

