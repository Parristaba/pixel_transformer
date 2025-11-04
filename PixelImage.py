import numpy as np
from PIL import Image


class PixelImage:
    def __init__(self, image_path, target_size=(512, 512)):
        self.path = image_path
        self.target_size = target_size
        # H x W x 3 array
        self.image_array = None         
        # N x 3 array
        self.flattened_pixels = None    
        # Original shape of the image
        self.original_shape = None      
        
        self._load_and_process()


    def _load_and_process(self):
        try:
            img = Image.open(self.path).convert('RGB')
            img = img.resize(self.target_size)

            image_array = np.array(img)
            self.image_array = image_array

            height, width, _ = image_array.shape
            self.original_shape = (height, width)

            # Flatten for k-NN
            self.flattened_pixels = image_array.reshape((height * width, 3))

        except FileNotFoundError:
            print(f"Error: Image not found at {self.path}")

    def _reshape_pixels(self, new_colors):
        height, width = self.original_shape
        if new_colors.shape[0] != height * width:
            raise ValueError("New colors array size does not match target image dimensions.")
            
        return new_colors.reshape(height, width, 3)