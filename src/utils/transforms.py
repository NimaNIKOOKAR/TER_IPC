# src/utils/transforms.py

from typing import Any, Dict, List
import numpy as np
from monai.transforms import MapTransform

class RemoveKeysd(MapTransform):
    """
    Supprime les clés non désirées du dictionnaire de données.
    """
    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            d.pop(key, None)
        return d


def combine_labels(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Combine les masques de labels en un seul label multi-classes.
    """
    label = np.zeros_like(data["label_lymph"], dtype=np.int64)
    label[data["label_lymph"] > 0] = 1
    label[data["label_subcar"] > 0] = 2
    label[data["label_azygos"] > 0] = 3
    label[data["label_esophagus"] > 0] = 4
    data["label"] = label
    return data