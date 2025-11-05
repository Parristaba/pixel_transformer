# PixelForge: Algorithmic Image Remapping
![Project Demo](Outputs/pixel_remap.gif)

---

## Project Overview

**PixelForge** is a Python-based project that demonstrates advanced image processing and data visualization by performing a complete **pixel-level transformation** between two images. It takes the **color profile** of a source image and imposes it onto the **spatial structure** of a target image, creating a new, hybrid artwork.

The project's highlight is a **spatially-aware video transition** that visualizes the movement of every single pixel from its starting location to its final position, proving the underlying sorting algorithm.

| Feature | Description |
| :--- | :--- |
| **Perceptual Sorting** | Uses **L*a*b* color space** for sorting pixels by perceived luminosity (L*), ensuring superior color fidelity compared to standard RGB/Grayscale methods. |
| **Guaranteed 1:1 Mapping** | Implements the **Brightness Sorting Algorithm** to ensure a perfect one-to-one pixel swap, overcoming the **index collision flaw** inherent in simpler methods like k-Nearest Neighbors. |
| **Spatially-Coherent Easing** | The transition video uses a custom non-linear easing function where pixel movement is dictated by **travel distance** and **start time** to create a realistic, flowing visual reveal. |
| **Pipeline Engineering** | Uses `joblib`, `Pillow`, and `OpenCV` to create a robust, modular pipeline that generates high-quality MP4 and web-friendly GIF output. |

---

## Technical Implementation & Architecture

### Algorithmic Breakdown: The Flaw and The Fix

The initial approach to pixel remapping often involves k-NN, which fails due to **index collision** (multiple source pixels mapping to the same target location), leaving holes in the final image.

PixelForge solves this by enforcing a **guaranteed one-to-one assignment** via L* Sorting:

1.  **Target Order:** The Target image pixels are sorted by L* to define the final **position order** ($P_{end}$).
2.  **Source Order:** The Source image pixels are sorted by L* to define the **color order** ($C_{source}$).
3.  **Assignment:** The $i$-th brightest $C_{source}$ pixel is assigned to the $i$-th brightest $P_{end}$ position, ensuring every pixel is used exactly once.

### Advanced Video Easing Logic

The final seamless video transition is achieved by defining a unique movement profile for every pixel:

1.  **Start Time ($\mathbf{T}_{start}$):** Defined by the pixel's **Normalized Travel Distance**. Pixels with the shortest physical distance to travel begin their journey earliest, logically forming the image structure first.
2.  **Movement Duration ($\mathbf{D}_{move}$):** Calculated proportionally to the pixel's **Euclidean travel distance**. Short-distance movers have a shorter duration, reinforcing the realism.
3.  **Time Scaling:** The entire process is numerically scaled to guarantee that the longest-moving pixel finishes precisely on the final frame of the transition, eliminating any abrupt "popping" and ensuring a smooth flow into the static target image.

---

## Project Structure & Usage

### Core Modules

* **`PixelImage.py`**: Handles image loading, resizing, and array flattening (`H x W x 3` to `N x 3`).
* **`utils.py`**: Houses the common helper function for **`calculate_luminosity`** using the L*a*b* color space.
* **`Brightness_Sorting.py`**: **Preparation Script.** Calculates and saves the Target image's luminosity sort order (`target_brightness_order.joblib`).
* **`generate_pixel_animation.py` (Main Script)**: Coordinates the entire visualization pipeline, generates the video, converts the GIF, and launches the video player.

### Prerequisites

```bash
pip install numpy pillow scikit-image joblib opencv-python imageio
```

#### 1. Prepare Target Data
Use a target image (e.g., target.jpg) to establish the structural map.

```bash
python Brightness_Sorting.py
# Creates TargetData/target_brightness_order.joblib and TargetData/target_shape.joblib
```

#### 2. Run Transformation & Visualization
Use a source image (e.g., source.jpg) to generate the output video and GIF.

```bash
python generate_pixel_animation.py
# Output: Outputs/pixel_remap.mp4 and Outputs/pixel_remap.gif
# Automatically launches the OpenCV video player.
```
