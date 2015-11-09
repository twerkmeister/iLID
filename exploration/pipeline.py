import generate_spectrograms
import time
import datetime
import os
import sys
import argparse
import classify
import numpy as np


def create_timestamp():
 ts = time.time()
 return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')

def pipeline(audio_file):
  timestamp = create_timestamp()
  spectrogram_dir = "%s_%s_spectrograms" % (timestamp, audio_file)

  if not os.path.exists(spectrogram_dir):
        os.mkdir(spectrogram_dir)

  if not os.path.isdir(spectrogram_dir):
        sys.exit("%s is not a directory." % spectrogram_dir)

  generate_spectrograms.create_and_write_spectrograms(os.path.dirname(audio_file), os.path.basename(audio_file), spectrogram_dir)

  spectrograms = os.listdir(spectrogram_dir)
  spectrograms = [os.path.join(spectrogram_dir, spectrogram) for spectrogram in spectrograms]
  results = [classify.classify(spectrograms)][0]

  print results
  print np.average(results, axis=0).argmax()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--in', dest='input', help='Input wav file to classify')
  args = parser.parse_args()

  if not os.path.exists(args.input):
    sys.exit("%s does not exist!" % args.input)
  
  pipeline(args.input)
