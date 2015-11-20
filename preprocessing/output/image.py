import cv2 as cv
import os

def save(filename, image, output_path=None):
  dir_name = os.path.dirname(filename)
  print image
  
  if output_path and dir_name != output_path:
    filename = os.path.join(output_path, os.path.basename(filename))

  #change extension
  filename_split = os.path.splitext(filename)
  filename = "".join([filename_split[0], ".png"])

  print filename
  cv.imwrite(filename, image)
  return filename

def show(f, image):
  cv.imshow(f, image)