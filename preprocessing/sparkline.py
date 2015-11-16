from pyspark import SparkContext
from util import *
import scipy.io.wavfile as wav
import sys

def main(files):
  sc = SparkContext("local", "sparkline", )
  samplerate_counts = (
    sc.parallelize(files)
    .map(lambda f: wav.read(f))
    .map(lambda samplerate_and_signal: (samplerate_and_signal[0], 1))
    .reduceByKey(lambda a,b: a + b))

  print samplerate_counts.collect()

if __name__ == '__main__':

  args = argparser.parse()
  files = filecollector.collect(args.input_path)
  print files
  main(files)