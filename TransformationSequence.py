import joblib
import numpy as np
from PIL import Image
from PixelImage import PixelImage
from utils import calculate_luminosity
import cv2
import os
import shutil
import imageio

SOURCE_IMAGE_PATH = 'SourceData/test.jpg'
ORDER_INPUT_PATH = 'TargetData/target_brightness_order.joblib'
SHAPE_INPUT_PATH = 'TargetData/target_shape.joblib'
VIDEO_OUTPUT_PATH = 'Outputs/pixel_remap.mp4'
GIF_OUTPUT_PATH = 'Outputs/pixel_remap.gif'
FRAME_OUTPUT_DIR = 'temp_animation_frames/'
IMAGE_SIZE = (1024, 1024)

TRANSITION_SECONDS = 7
STILL_SECONDS = 2
FPS = 30
TRANSITION_FRAMES = TRANSITION_SECONDS * FPS
STILL_FRAMES = STILL_SECONDS * FPS
TOTAL_FRAMES = (STILL_FRAMES * 2) + TRANSITION_FRAMES

MIN_DURATION_RATIO = 0.05
MAX_DURATION_RATIO = 0.50


def generate_coordinates(H, W):
    """Return an (H*W, 2) array of (x, y) integer coordinates for a grid of height H and width W."""
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    return np.vstack([x_coords.ravel(), y_coords.ravel()]).T


def generate_pixel_animation():
    """Load source and target data, compute spatial remapping with distance-based easing,
    render transition frames into an MP4, and play the resulting video.
    """
    if os.path.exists(FRAME_OUTPUT_DIR):
        shutil.rmtree(FRAME_OUTPUT_DIR)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    target_order_indices = joblib.load(ORDER_INPUT_PATH)
    H, W = joblib.load(SHAPE_INPUT_PATH)

    source_img_obj = PixelImage(SOURCE_IMAGE_PATH, target_size=IMAGE_SIZE)
    source_colors = source_img_obj.flattened_pixels.astype(np.uint8)
    N = source_colors.shape[0]

    P_Start = generate_coordinates(H, W)

    source_luminosity = calculate_luminosity(source_colors)
    source_sort_indices = np.argsort(source_luminosity)

    P_End_Indices = np.empty_like(source_sort_indices)
    P_End_Indices[source_sort_indices] = target_order_indices
    P_End = P_Start[P_End_Indices]

    print("Calculating movement distances and spatial durations...")

    distances = np.linalg.norm(P_End - P_Start, axis=1)
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    duration_ratio_raw = MIN_DURATION_RATIO + normalized_distances * (MAX_DURATION_RATIO - MIN_DURATION_RATIO)
    T_start_raw = normalized_distances

    T_finish = T_start_raw + duration_ratio_raw
    MAX_FINISH_TIME = np.max(T_finish)

    T_start = T_start_raw / MAX_FINISH_TIME
    duration_ratio_scaled = duration_ratio_raw / MAX_FINISH_TIME

    final_colors = np.empty_like(source_colors, dtype=np.uint8)
    final_colors[target_order_indices] = source_colors[source_sort_indices]
    Target_Image_Array = final_colors.reshape(H, W, 3)
    Source_Image_Array = source_img_obj.image_array

    print("Initializing VideoWriter...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, FPS, (W, H))

    print(f"-> Adding Source Still Frames ({STILL_SECONDS}s)...")
    for _ in range(STILL_FRAMES):
        video_writer.write(cv2.cvtColor(Source_Image_Array, cv2.COLOR_RGB2BGR))

    print(f"-> Adding Spatially-Coherent Transition Frames ({TRANSITION_SECONDS}s)...")
    for frame_idx in range(TRANSITION_FRAMES):
        T_frame_progress = frame_idx / (TRANSITION_FRAMES - 1)
        time_elapsed_since_start = T_frame_progress - T_start
        raw_alpha_i = time_elapsed_since_start / duration_ratio_scaled
        alpha_i = np.clip(raw_alpha_i, 0.0, 1.0)

        P_frame_float = (1 - alpha_i[:, np.newaxis]) * P_Start + alpha_i[:, np.newaxis] * P_End
        P_frame_int = np.round(P_frame_float).astype(int)

        frame_array = np.full((H, W, 3), 0, dtype=np.uint8)

        x_coords = np.clip(P_frame_int[:, 0], 0, W - 1)
        y_coords = np.clip(P_frame_int[:, 1], 0, H - 1)

        frame_array[y_coords, x_coords] = source_colors

        video_writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))

    print(f"-> Adding Target Still Frames ({STILL_SECONDS}s)...")
    for _ in range(STILL_FRAMES):
        video_writer.write(cv2.cvtColor(Target_Image_Array, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"\n✅ Video generation complete. Saved to: {VIDEO_OUTPUT_PATH}")

    try:
        print("Converting MP4 to GIF...")
        cap_gif = cv2.VideoCapture(VIDEO_OUTPUT_PATH)
        if not cap_gif.isOpened():
            print("Error: could not open generated MP4 for GIF conversion.")
        else:
            # Downscale GIF if very wide to keep file size reasonable
            GIF_MAX_WIDTH = 512
            scale = 1.0
            if W > GIF_MAX_WIDTH:
                scale = GIF_MAX_WIDTH / float(W)
                gif_size = (int(W * scale), int(H * scale))
            else:
                gif_size = (W, H)

            writer = imageio.get_writer(GIF_OUTPUT_PATH, mode='I', duration=1.0 / FPS)
            while True:
                ret, frame = cap_gif.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale != 1.0:
                    rgb = cv2.resize(rgb, gif_size, interpolation=cv2.INTER_AREA)
                writer.append_data(rgb)
            writer.close()
            cap_gif.release()
            print(f"✅ GIF saved to: {GIF_OUTPUT_PATH}")
    except Exception as e:
        print("GIF conversion failed:", e)

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

            if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Final presentation complete.")


if __name__ == "__main__":
    generate_pixel_animation()