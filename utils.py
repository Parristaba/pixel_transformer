from skimage.color import rgb2lab

def calculate_luminosity(rgb_array):
    """
    Converts N x 3 RGB array (uint8) to L*a*b* space and returns the Luminosity (L*) channel.
    
    L* is the perceptual measure of brightness, ranging from 0 to 100.
    """
    rgb_float = rgb_array.astype(float)
    lab_array = rgb2lab(rgb_float)
    
    return lab_array[:, 0]