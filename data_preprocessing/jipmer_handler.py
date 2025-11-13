"""
jipmer_handler.py

Handler for the JIPMER CT scan dataset format, supporting multiple imaging phases (Arterial, Portal, Venous). Extracts subject lists and validates dataset structure for preprocessing pipelines.
"""
from typing import Dict, List

from data_preprocessing.handler import DatasetHandler

PHASE_SUFFIXES = {
    "Arterial": "A",
    "Portal": "P",
    "Venous": "V"
}

# This handler is now deprecated for vessel preprocessing only. Remove liver/tumor logic and leave as a placeholder if not used.
class JIPMERHandler(DatasetHandler):
    """
    Placeholder/Deprecated: JIPMER dataset is not used for vessel segmentation pipeline.
    """
    def __init__(self, dataset_path: str, phase: str = "Arterial"):
        super().__init__(dataset_path, "jipmer")
    def validate_dataset(self) -> bool:
        return False  # Not used
    def get_subject_list(self):
        return []