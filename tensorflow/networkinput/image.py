from scipy.ndimage import imread
import numpy as np

def read_png(file_path, mode):
    image = imread(file_path, mode=mode)
    if mode == "L":
        #Adding third dimension to fit channel structure
        image = np.reshape(image, image.shape+(1,))
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    assert(len(image.shape) >= 3)
    return image