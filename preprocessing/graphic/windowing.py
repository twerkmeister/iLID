import numpy as np
import math

def sliding(image, windowsize, stride):
  """creates an generator of sliding window along the width of an image

  Args:
      image (2 or 3 dimensional numpy array): the image the sliding windows are created for
      windowsize (int): width of the sliding window
      stride (int): stepsize of the sliding

  Returns:
      generator containing 

  """
  assert len(image.shape) > 1 and len(image.shape) < 4

  image_height = image.shape[0]
  image_width = image.shape[1]
  
  has_channels = len(image.shape) > 2
  image_channels = None
  if has_channels:
    image_channels = image.shape[2]

  number_windows = int(math.ceil(image_width / float(stride)))
  print number_windows
  
  for i in range(number_windows):
    
    window = image[:,i*stride:i*stride+windowsize]
    window_height = window.shape[0]
    window_width = window.shape[1]
    
    assert window_height == image_height
    
    if window_width != windowsize:
      missing_window_width = windowsize - window_width
      padding = None
      if has_channels:
        padding = np.zeros((window_height, missing_window_width, image_channels))
      else:
        padding = np.zeros((window_height, missing_window_width))
      window = np.append(window, padding, axis=1)

    yield window