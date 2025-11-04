import joblib
import numpy as np
from PIL import Image
from PixelImage import PixelImage  # Import your class
from utils import calculate_luminosity

SOURCE_IMAGE_PATH = 'President_Barack_Obama.jpg'
ORDER_INPUT_PATH = 'TargetData/target_brightness_order.joblib'
SHAPE_INPUT_PATH = 'TargetData/target_shape.joblib'
OUTPUT_IMAGE_PATH = 'remapped_result_sorted.png'
IMAGE_SIZE = (1024, 1024)

def remap_pixels_sorted():
    """Sort source pixels by L* and place them into target positions."""
    target_order_indices = joblib.load(ORDER_INPUT_PATH)
    target_shape = joblib.load(SHAPE_INPUT_PATH)

    source_img_obj = PixelImage(SOURCE_IMAGE_PATH, target_size=IMAGE_SIZE)
    source_colors = source_img_obj.flattened_pixels
    if source_colors is None:
        print("Remapping aborted: failed to load source image.")
        return

    source_brightness = calculate_luminosity(source_colors)
    source_sort_indices = np.argsort(source_brightness)
    sorted_source_colors = source_colors[source_sort_indices].astype(np.uint8)

    final_colors = np.empty_like(source_colors, dtype=np.uint8)
    final_colors[target_order_indices] = sorted_source_colors

    H, W = target_shape
    final_image_array = final_colors.reshape(H, W, 3)
    final_image = Image.fromarray(final_image_array, 'RGB')
    final_image.save(OUTPUT_IMAGE_PATH)

    print(f"Image remapped and saved to: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    remap_pixels_sorted()