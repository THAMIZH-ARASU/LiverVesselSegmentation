"""
handler.py

Base class for dataset handlers in CT scan segmentation preprocessing. Defines the interface for dataset validation and subject list extraction for different dataset formats (e.g., MSD, JIPMER).
"""
from pathlib import Path
from typing import Dict, List
import pydicom
import nibabel as nib
import numpy as np


class DatasetHandler:
    """
    Base class for dataset handling in preprocessing pipelines.

    Attributes:
        dataset_path (Path): Path to the dataset root.
        dataset_type (str): Type of dataset (e.g., 'medical_decathlon', 'jipmer').
    """
    
    def __init__(self, dataset_path: str, dataset_type: str):
        """
        Initialize the DatasetHandler.

        Args:
            dataset_path (str): Path to the dataset root.
            dataset_type (str): Type of dataset.
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        
    def get_subject_list(self) -> List[Dict[str, str]]:
        """
        Get a list of subjects with their file paths for images and labels.

        Returns:
            List[Dict[str, str]]: List of subject metadata dicts.
        """
        raise NotImplementedError
        
    def validate_dataset(self) -> bool:
        """
        Validate the dataset structure (e.g., required directories/files exist).

        Returns:
            bool: True if dataset is valid, False otherwise.
        """
        raise NotImplementedError

class DircadbVesselHandler(DatasetHandler):
    """
    Handler for 3DIRCADB dataset. Converts DICOM (CT+mask) to NIfTI; extracts only hepatic vessel mask and image.
    """
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, "dircadb_vessel")
        self.patient_dirs = [p for p in self.dataset_path.iterdir() if p.is_dir() and p.name.startswith('3Dircadb1.')]
    def validate_dataset(self) -> bool:
        if not self.patient_dirs:
            print(f"No patient subfolders found in {self.dataset_path}")
            return False
        return True
    def get_subject_list(self) -> list:
        """
        For each patient:
        - unzip PATIENT_DICOM.zip, MASKS_DICOM.zip if not already
        - convert PATIENT_DICOM/*.dcm to one CT NIfTI
        - convert MASKS_DICOM/portalvein/*.dcm or venoussystem/*.dcm to vessel NIfTI mask
        - return [{subject_id, image, vessel_label}]
        """
        import shutil, os
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for DICOM-to-NIfTI conversion")
        subjects = []
        for patient in self.patient_dirs:
            # Unzip if needed
            unzip_dir = lambda zipfile, outdir: shutil.unpack_archive(zipfile, outdir) if not (patient/outdir).exists() else None
            unzip_dir(patient/'PATIENT_DICOM.zip', 'PATIENT_DICOM')
            unzip_dir(patient/'MASKS_DICOM.zip', 'MASKS_DICOM')
            patient_ct_dir = patient/'PATIENT_DICOM'
            mask_root = patient/'MASKS_DICOM'
            # Strictly select hepatic vessels
            vessel_dir = None
            possible = [mask_root/'hepaticvessel', mask_root/'HepaticVessels', mask_root/'hepatic_vessels']
            for vpath in possible:
                if vpath.exists():
                    vessel_dir = vpath
                    break
            if vessel_dir is None:
                # fallback (log warning) to portalvein, venoussystem
                if (mask_root/'portalvein').exists():
                    vessel_dir = mask_root/'portalvein'
                    print(f"Warning: Using portalvein mask for {patient.name}")
                elif (mask_root/'venoussystem').exists():
                    vessel_dir = mask_root/'venoussystem'
                    print(f"Warning: Using venoussystem mask for {patient.name}")
                else:
                    print(f"Error: No hepatic vessel/portalvein/venoussystem mask in {patient.name}")
                    continue
            if not patient_ct_dir.exists() or not vessel_dir.exists():
                print(f"Missing DICOM CT or vessel mask for {patient.name}")
                continue
            # DICOM to NIfTI CT
            ct_nifti = patient / 'ct.nii.gz'
            if not ct_nifti.exists():
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(str(patient_ct_dir))
                if not dicom_files:
                    print(f"Error: No DICOM CT slices found for {patient.name}")
                    continue
                reader.SetFileNames(dicom_files)
                ct_image = reader.Execute()
                sitk.WriteImage(ct_image, str(ct_nifti))
            # DICOM to NIfTI Vessel mask
            vessel_nifti = patient / 'vessel.nii.gz'
            if not vessel_nifti.exists():
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(str(vessel_dir))
                if not dicom_files:
                    print(f"Error: No DICOM vessel mask slices found for {patient.name}")
                    continue
                reader.SetFileNames(dicom_files)
                mask = reader.Execute()
                sitk.WriteImage(mask, str(vessel_nifti))
            subjects.append({
                'subject_id': patient.name,
                'image': str(ct_nifti),
                'vessel_label': str(vessel_nifti)
            })
        return subjects