import numpy as np
import caffe
import os
import sys

predict_path = os.path.abspath(os.path.join('../preprocessing'))
sys.path.append(predict_path)

import graphic
import output
import audio

def flatmap(f, seq):
  return [f(s) for s in seq]

def read_wav(f):
  samplerate, signal = wav.read(f)
  #if len(signal.shape) > 1:
  #  signal = signal[:,0]
  f = filename.truncate_extension(filename.clean(f))
  return (f, signal, samplerate)

def apply_melfilter(f, signal, samplerate):
  filterbank_energies = audio.melfilterbank.logfilter(samplerate, signal, winlen=0.00833, winstep=0.00833, nfilt=39, lowfreq=0, preemph=1.0)
  #print f, samplerate, filterbank_energies.shape
  return (f, filterbank_energies)

def generate_spectrograms(f, signal, samplerate):
  Sxx = audio.spectrogram.spectrogram_cutoff(samplerate, signal, winlen=0.00833, winstep=0.00833)
  return (f, Sxx)

def sliding_audio(f, signal, samplerate):
  for window_name, window in audio.windowing.sliding_with_filename(f, signal, samplerate, 5, 5, 0.6):
    yield (window_name, window, samplerate)

def downsample(f, signal, samplerate):
  target_samplerate = 16000
  downsampled_signal, downsampled_samplerate = audio.resample.downsample(signal, samplerate, target_samplerate)
  return (f, downsampled_signal, downsampled_samplerate)

def predict(sound_file, prototxt, model, output_path):

  if not os.path.isdir(output_path):
    os.mkdir(output_path)

  (imageread_wav(sound_file)
    .flatMap(lambda (f, signal, samplerate): sliding_audio(f, signal, samplerate))
    .map(lambda (f, signal, samplerate): downsample(f, signal, samplerate))
    .map(lambda (f, signal, samplerate): apply_melfilter(f, signal, samplerate))
    .map(lambda (f, image): (f, graphic.colormapping.to_grayscale(image, bytes=True)))
    .map(lambda (f, image): (f, graphic.histeq.histeq(image)))
    .map(lambda (f, image): (f, graphic.histeq.clamp_and_equalize(image)))
    .map(lambda (f, image): (f, graphic.windowing.cut_or_pad_window(image, window_size)))
    .map(lambda (f, image): output.image.save(f, image, output_path))
  )

  caffe.set_mode_cpu()
  net = caffe.Classifier(prototxt, model,
                         #image_dims=(224, 224)
                         #channel_swap=(2,1,0))
                         #raw_scale=255)
                         caffe.TEST
                        )

  input_images = [caffe.io.load_image(image_file) for image_file in image_files]
  #plt.imshow(input_image)

  prediction = net.forward_all(data=input_images)

  #prediction = net.predict(input_images, False)  # predict takes any number of images, and formats them for the Caffe net automatically
  #print 'prediction shape:', prediction[0].shape
  #print 'predicted class:', prediction[0].argmax()
  return prediction