import joblib
import numpy as np
from PIL import Image
from PixelImage import PixelImage 
from utils import calculate_luminosity
import cv2 
import os
import shutil

# --- CONFIGURATION ---
SOURCE_IMAGE_PATH = 'malak.jpg'
ORDER_INPUT_PATH = 'TargetData/target_brightness_order.joblib' 
SHAPE_INPUT_PATH = 'TargetData/target_shape.joblib'
VIDEO_OUTPUT_PATH = 'pixel_remap_animation_spatial_final.mp4' # Final filename
FRAME_OUTPUT_DIR = 'temp_animation_frames/'
IMAGE_SIZE = (1024, 1024) 

# Animation & Sequence Settings
TRANSITION_SECONDS = 7     # ~1.43x speedup
STILL_SECONDS = 2
FPS = 30
TRANSITION_FRAMES = TRANSITION_SECONDS * FPS
STILL_FRAMES = STILL_SECONDS * FPS
TOTAL_FRAMES = (STILL_FRAMES * 2) + TRANSITION_FRAMES

# Distance-Based Easing Parameters
MIN_DURATION_RATIO = 0.05
MAX_DURATION_RATIO = 0.50 

# --- CORE UTILITIES ---

def generate_coordinates(H, W):
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    return np.vstack([x_coords.ravel(), y_coords.ravel()]).T

# --- MAIN GENERATOR FUNCTION ---

def generate_pixel_animation():
    
    # 0. Setup and Load Data
    if os.path.exists(FRAME_OUTPUT_DIR):
        shutil.rmtree(FRAME_OUTPUT_DIR)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
    
    target_order_indices = joblib.load(ORDER_INPUT_PATH)
    H, W = joblib.load(SHAPE_INPUT_PATH)
    
    source_img_obj = PixelImage(SOURCE_IMAGE_PATH, target_size=IMAGE_SIZE)
    source_colors = source_img_obj.flattened_pixels.astype(np.uint8)
    N = source_colors.shape[0]

    # 1. Define Start/End Points
    P_Start = generate_coordinates(H, W)
    
    source_luminosity = calculate_luminosity(source_colors)
    source_sort_indices = np.argsort(source_luminosity)
    
    P_End_Indices = np.empty_like(source_sort_indices)
    P_End_Indices[source_sort_indices] = target_order_indices
    P_End = P_Start[P_End_Indices]
    
    # 2. Calculate Distance-Based Movement Properties
    print("Calculating movement distances and spatial durations...")
    
    distances = np.linalg.norm(P_End - P_Start, axis=1)
    
    # Normalize distances (0.0 to 1.0)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    # Duration Ratio: Set duration proportional to distance
    duration_ratio_raw = MIN_DURATION_RATIO + normalized_distances * (MAX_DURATION_RATIO - MIN_DURATION_RATIO)
    
    # T_start_raw: Time when movement begins (short distance = early start)
    T_start_raw = normalized_distances
    
    # --- CRITICAL FIX: TIME SCALING ---
    # Calculate the finish time for every pixel: T_finish = T_start_raw + duration_ratio_raw
    T_finish = T_start_raw + duration_ratio_raw
    
    # Find the maximum finish time (should be slightly > 1.0)
    MAX_FINISH_TIME = np.max(T_finish) 
    
    # Scale both T_start and duration_ratio so that the MAX_FINISH_TIME aligns perfectly with 1.0.
    T_start = T_start_raw / MAX_FINISH_TIME
    duration_ratio_scaled = duration_ratio_raw / MAX_FINISH_TIME 

    # Pre-calculate still image arrays
    final_colors = np.empty_like(source_colors, dtype=np.uint8)
    final_colors[target_order_indices] = source_colors[source_sort_indices]
    Target_Image_Array = final_colors.reshape(H, W, 3)
    Source_Image_Array = source_img_obj.image_array

    # 3. Initialize Video Writer
    print("Initializing VideoWriter...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, FPS, (W, H))

    
    # --- Sequence: 1. Hold Source Image ---
    print(f"-> Adding Source Still Frames ({STILL_SECONDS}s)...")
    for _ in range(STILL_FRAMES):
        video_writer.write(cv2.cvtColor(Source_Image_Array, cv2.COLOR_RGB2BGR))
    
    # --- Sequence: 2. Distance-Eased Transition ---
    print(f"-> Adding Spatially-Coherent Transition Frames ({TRANSITION_SECONDS}s)...")
    
    for frame_idx in range(TRANSITION_FRAMES):
        T_frame_progress = frame_idx / (TRANSITION_FRAMES - 1)
        
        # Calculate individual alpha (progress)
        time_elapsed_since_start = T_frame_progress - T_start
        
        # Scale time elapsed by the inverse of the individual duration ratio
        raw_alpha_i = time_elapsed_since_start / duration_ratio_scaled # Using SCALED duration
        
        # Clip the result to ensure it stays between 0.0 (not started) and 1.0 (finished).
        alpha_i = np.clip(raw_alpha_i, 0.0, 1.0)

        # Calculate interpolated position 
        P_frame_float = (1 - alpha_i[:, np.newaxis]) * P_Start + alpha_i[:, np.newaxis] * P_End
        P_frame_int = np.round(P_frame_float).astype(int)
        
        # Create Frame Canvas
        frame_array = np.full((H, W, 3), 0, dtype=np.uint8)
        
        # Draw Pixels onto Canvas 
        x_coords = np.clip(P_frame_int[:, 0], 0, W - 1)
        y_coords = np.clip(P_frame_int[:, 1], 0, H - 1)
        
        frame_array[y_coords, x_coords] = source_colors
        
        video_writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
        
    # --- Sequence: 3. Hold Target Image ---
    print(f"-> Adding Target Still Frames ({STILL_SECONDS}s)...")
    for _ in range(STILL_FRAMES):
        video_writer.write(cv2.cvtColor(Target_Image_Array, cv2.COLOR_RGB2BGR))

    # Finalize video file and play
    video_writer.release()
    print(f"\n✅ Video generation complete. Saved to: {VIDEO_OUTPUT_PATH}")

    # 4. Play Video (OpenCV Player)
    print("📺 Displaying video player (Press 'q' to close window)...")
    
    cap = cv2.VideoCapture(VIDEO_OUTPUT_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file for playback.")
        return

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Distance-Based Pixel Remap', frame)
            
            if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"🎉 Final presentation complete.")

if __name__ == "__main__":
    generate_pixel_animation()