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