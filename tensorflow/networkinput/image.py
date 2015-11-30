from scipy.ndimage import imread
import numpy as np

def read_png(file_path, mode):
    image = imread(file_path, mode=mode)
    if mode == "L":
        #Adding third dimension to fit channel structure
        image = np.reshape(image, image.shape+(1,))
    assert(len(image.shape) >= 3)
    return image