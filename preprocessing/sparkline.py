from pyspark import SparkContext
from util import *
import scipy.io.wavfile as wav

def main(files):
  sc = SparkContext("local", "sparkline")
  samplerate_counts = (
    sc.parallelize(files)
    .map(lambda f: wav.read(f))
    .map(lambda samplerate,signal: (samplerate, 1))
    .reduceByKey(lambda a, b: a+b))

  print(samplerate_counts.collect())

if __name__ == '__main__':
  args = argparser.parse()
  main(filecollector.collect(args.input_path))