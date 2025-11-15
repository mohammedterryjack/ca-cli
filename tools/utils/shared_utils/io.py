

from pathlib import Path
from PIL import Image
import numpy as np
from numpy import array

def save_evolution(values: list[list[int]], path: Path) -> None:
    arr = array(values, dtype="uint8")
    N = arr.max()
    if N == 0:
        img_data = arr.astype("uint8")  # all zeros, no scaling needed
    else:
        img_data = (arr / N * 255).astype("uint8")
    img = Image.fromarray(img_data, mode="L")
    img.save(str(path))




def load_evolution(image_path: Path) -> np.ndarray:
    """
    Load a CA evolution saved as an image (grayscale) and convert back to integer states.
    """
    img = Image.open(str(image_path)).convert("L")  # ensure grayscale
    arr = np.array(img, dtype=np.float32)          # float for scaling
    if arr.max() == 0:
        return arr.astype(int)
    else:
        return (arr >= 128).astype(int)

