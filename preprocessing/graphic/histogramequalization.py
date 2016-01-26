import numpy as np

def histeq(image, number_bins=256):
   #Adapted from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
   #get image histogram
   histogram,bins = np.histogram(image.flatten(), number_bins, normed=True)
   cdf = histogram.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   normalized_image = np.interp(image.flatten(), bins[:-1], cdf)

   return normalized_image.reshape(image.shape)

def clamp_and_equalize(image):
   # Clamp and normalize the remaining histogram
   # Similar to Levels Filter in Photoshop etc

   min_val = 100 # clamp ca. 40% of black values
   max_val = image.max()

   LUT = np.zeros((256,)) # Look up table

   # clamp all values smaller/greater than min/max
   LUT[:min_val] = 0
   LUT[max_val:] = 255

   # re-distribute all values between min and max
   factor = (255.0 / (max_val - min_val))
   base = 0

   for i in range(min_val, max_val):
    LUT[i] = base
    base += factor

   # apply the Lookup Table
   height, width = image.shape

   for i in np.arange(height):
    for j in np.arange(width):
      image[i][j] = LUT[image[i][j]]

   # return np.take(LUT, image)
   return image