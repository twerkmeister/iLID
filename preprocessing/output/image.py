import cv2 as cv
import os

def save(filename, images, output_path=None):
  dir_name = os.path.dirname(filename)
  #print images
  
  if output_path and dir_name != output_path:
    filename = os.path.join(output_path, os.path.basename(filename))

  #change extension
  filename_split = os.path.splitext(filename)
  for i in range(len(images)):
    counter = "_%02d" % i 
    filename = "".join([filename_split[0], counter, ".png"])

    print filename
    cv.imwrite(filename, images[i])
    
  return filename

def show(f, image):
  cv.imshow(f, image)