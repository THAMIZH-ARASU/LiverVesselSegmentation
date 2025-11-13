"""
msd_handler.py

Handler for the Medical Segmentation Decathlon (MSD) dataset format. Extracts subject lists and validates dataset structure for preprocessing pipelines.
"""
from typing import Dict, List
from data_preprocessing.handler import DatasetHandler


class HepaticVesselHandler(DatasetHandler):
    """
    Handler for Hepatic Vessel Segmentation dataset format (MSD Hepatic Vessels).
    Handles .nii/.nii.gz files containing vessel masks (label==1).
    """
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, "hepatic_vessel")
        self.images_dir = self.dataset_path / "imagesTr"
        self.labels_dir = self.dataset_path / "labelsTr"

    def validate_dataset(self) -> bool:
        if not self.images_dir.exists() or not self.labels_dir.exists():
            print(f"Error: Expected directories 'imagesTr' and 'labelsTr' not found in {self.dataset_path}")
            return False
        return True

    def get_subject_list(self) -> List[Dict[str, str]]:
        import nibabel as nib
        subjects = []
        for img_file in self.images_dir.glob("*.nii*"):
            subject_id = img_file.stem.replace(".nii", "")
            label_file = self.labels_dir / img_file.name
            if label_file.exists():
                # Check vessel label exists (label == 1)
                label_img = nib.load(str(label_file))
                arr = label_img.get_fdata()
                if (arr == 1).any():
                    subjects.append({
                        "subject_id": subject_id,
                        "image": str(img_file),
                        "vessel_label": str(label_file)
                    })
                else:
                    print(f"Warning: No vessel label in {label_file.name}")
            else:
                print(f"Warning: Label file not found for {img_file.name}")
        return subjects