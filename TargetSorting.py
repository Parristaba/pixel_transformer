import joblib
import numpy as np
from PixelImage import PixelImage
from utils import calculate_luminosity
import os


TARGET_IMAGE_PATH = 'efne.jpg'
ORDER_OUTPUT_PATH = 'TargetData/target_brightness_order.joblib'
SHAPE_OUTPUT_PATH = 'TargetData/target_shape.joblib'
IMAGE_SIZE = (1024, 1024)

def prepare_target_order():
    """Compute L* luminosity order for the target image and save indices and shape."""
    print(f"Loading target image: {TARGET_IMAGE_PATH}")
    target_img_obj = PixelImage(TARGET_IMAGE_PATH, target_size=IMAGE_SIZE)
    target_colors = target_img_obj.flattened_pixels
    if target_colors is None:
        print("Image load failed.")
        return

    target_luminosity = calculate_luminosity(target_colors)
    target_order_indices = np.argsort(target_luminosity)

    os.makedirs(os.path.dirname(ORDER_OUTPUT_PATH), exist_ok=True)
    print(f"Saving order to {ORDER_OUTPUT_PATH} and shape to {SHAPE_OUTPUT_PATH}")
    joblib.dump(target_order_indices, ORDER_OUTPUT_PATH)
    joblib.dump(target_img_obj.original_shape, SHAPE_OUTPUT_PATH)
    print("Saved.")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(ORDER_OUTPUT_PATH), exist_ok=True)
    
    prepare_target_order()