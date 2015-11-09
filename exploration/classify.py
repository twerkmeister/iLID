import numpy as np
import matplotlib.pyplot as plt
import caffe

def classify(IMAGE_FILES):

  # Set the right path to your model definition file, pretrained model weights,
  # and the image you would like to classify.
  MODEL_FILE = '../models/VGG_M_finetune/net_deploy.prototxt'
  PRETRAINED = '../models/VGG_M_finetune/vgg_m_finetune_iter_400001.caffemodel'


  caffe.set_mode_cpu()
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                         #mean=np.array([104,117,123]),
                         channel_swap=(2,1,0))
                         #raw_scale=255)
                         #mage_dims=(224, 224))
  
  input_images = [caffe.io.load_image(image_file) for image_file in IMAGE_FILES]
  #plt.imshow(input_image)

  prediction = net.predict(input_images, False)  # predict takes any number of images, and formats them for the Caffe net automatically
  #print 'prediction shape:', prediction[0].shape
  #print 'predicted class:', prediction[0].argmax()
  return prediction