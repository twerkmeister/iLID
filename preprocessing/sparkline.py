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
  filterbank_energies = audio.melfilterbank.logfilter(samplerate, signal)
  return (f, filterbank_energies)

def main(args):
  files = filecollector.collect(args.input_path)

  sc = SparkContext("local", "sparkline", )
  pipeline = (
    sc.parallelize(files)
    .map(lambda f: read_wav(f))
    .map(lambda (f, samplerate_and_signal): apply_melfilter(f, samplerate_and_signal[0], samplerate_and_signal[1]))
    .map(lambda (f, filterbank_energies): (f, graphic.colormapping.to_rgb(filterbank_energies, bytes=True)))
    .map(lambda (f, image): (f, list(graphic.windowing.sliding(image, 600, 600))))
    .map(lambda (f, images): output.image.save(f, images, args.output_path)))
  
  pipeline.collect()


if __name__ == '__main__':

  args = argparser.parse()
  main(args)