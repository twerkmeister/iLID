from pyspark import SparkContext
from util import *
import audio
import graphic
import output
import scipy.io.wavfile as wav
import sys

def read_wav(f):
  return (f, wav.read(f))

def apply_melfilter(f, samplerate, signal):
  filterbank_energies = audio.melfilterbank.logfilter(samplerate, signal, winlen=0.00833, winstep=0.00833, nfilt=39, lowfreq=0, preemph=1.0)
  return (f, filterbank_energies)

def main(args):
  window_size = 600
  files = filecollector.collect(args.input_path)

  sc = SparkContext("local", "sparkline")
  pipeline = (
    sc.parallelize(files, 4)
    .map(lambda f: read_wav(f))
    .map(lambda (f, samplerate_and_signal): (filename.truncate_extension(f), samplerate_and_signal))
    .map(lambda (f, samplerate_and_signal): apply_melfilter(f, samplerate_and_signal[0], samplerate_and_signal[1]))
    .map(lambda (f, filterbank_energies): (f, graphic.colormapping.to_grayscale(filterbank_energies, bytes=True)))
    .flatMap(lambda (f, image): list(graphic.windowing.sliding_with_filenames(f, image, window_size, window_size, 0.6)))
    .map(lambda (f, image): (f, graphic.histeq.histeq(image)))
    .map(lambda (f, image): (f, graphic.windowing.pad_window(image, window_size)))
    .map(lambda (f, image): output.image.save(f, image, args.output_path))
  )
  
  pipeline.collect()


if __name__ == '__main__':

  args = argparser.parse()
  main(args)