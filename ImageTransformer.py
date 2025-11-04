# remap_image_sorted.py

import joblib
import numpy as np
from PIL import Image
from PixelImage import PixelImage  # Import your class

# --- CONFIGURATION (UPDATED FOR SORTING) ---
SOURCE_IMAGE_PATH = 'guts.jpg' 
ORDER_INPUT_PATH = 'TargetData/target_brightness_order.joblib' 
SHAPE_INPUT_PATH = 'TargetData/target_shape.joblib'
OUTPUT_IMAGE_PATH = 'remapped_result_sorted.png'
IMAGE_SIZE = (512, 512) # MUST match the size used during preparation!

# --- HELPER FUNCTION (Same as used in preparation) ---
def calculate_brightness(rgb_array):
    """Calculates the perceived brightness (luminance) for an array of RGB pixels."""
    rgb_float = rgb_array.astype(float)
    return (
        0.299 * rgb_float[:, 0] +
        0.587 * rgb_float[:, 1] +
        0.114 * rgb_float[:, 2]
    )

def remap_pixels_sorted():
    """
    Loads the Source Image, sorts its pixels by brightness, and maps them 
    onto the predefined target positions. This guarantees a 1:1 assignment.
    """
    
    # 1. Load Target Order and Shape
    print("Loading target order indices and shape...")
    # This array defines the POSITION order (darkest target pixel location first)
    target_order_indices = joblib.load(ORDER_INPUT_PATH)
    target_shape = joblib.load(SHAPE_INPUT_PATH)
    
    # 2. Process Source Image
    print(f"Processing source image: {SOURCE_IMAGE_PATH}...")
    source_img_obj = PixelImage(SOURCE_IMAGE_PATH, target_size=IMAGE_SIZE)
    source_colors = source_img_obj.flattened_pixels
    
    if source_colors is None:
        print("Remapping aborted due to source image loading error.")
        return

    # 3. Sort Source Colors (The Core Transformation Logic)
    print("Sorting source colors by brightness...")
    
    # Calculate brightness for the Source image
    source_brightness = calculate_brightness(source_colors)
    
    # Get the indices that would sort the Source colors by brightness
    # This gives the ORDER of the Source colors from darkest (0) to lightest (N-1)
    source_sort_indices = np.argsort(source_brightness)
    
    # Use the sort indices to arrange the actual RGB colors of the Source image
    # The resulting array holds the colors in order of increasing brightness
    sorted_source_colors = source_colors[source_sort_indices].astype(np.uint8)
    
    # 4. Final Image Assembly (The 1:1 Assignment)
    print("Assembling the final image with 1:1 mapping...")
    N = source_colors.shape[0]
    
    # Initialize the final array. No need for full initialization here since 
    # every index will be written to exactly once.
    final_colors = np.empty_like(source_colors, dtype=np.uint8)
    
    # CRITICAL STEP: Assign the sorted Source colors to the Target positions.
    # target_order_indices[i] gives the POSITION in the final image.
    # sorted_source_colors[i] gives the COLOR to put there.
    final_colors[target_order_indices] = sorted_source_colors
    
    # 5. Reshape and Save
    H, W = target_shape
    final_image_array = final_colors.reshape(H, W, 3)
    
    # Convert back to PIL Image and save
    final_image = Image.fromarray(final_image_array, 'RGB')
    final_image.save(OUTPUT_IMAGE_PATH)
    
    print(f"\n✅ Image successfully remapped and saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    # --- IMPORTANT --- 
    # Run the prepare_target_order.py script first!
    # --- IMPORTANT ---
    remap_pixels_sorted()