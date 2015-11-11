import os
import subprocess
import argparse
import math
import sys
import numpy as np
from progressbar import ProgressBar

def create_and_write_spectrograms(root, sound_file, output_path):
    # Obtain audio file's length
    abs_path = os.path.join(root, sound_file)
    command = "soxi -D {0}".format(abs_path)
    audio_length = float(subprocess.check_output(command, shell=True))

    # Create overlapping slices + spectrogram for a single audio file
    for i in np.arange(0, math.floor(audio_length) - 0.5, 0.5):
        overlap_start = i
        overlap_end = overlap_start + 1

        filename = os.path.join(output_path, "{0}_{1}.png".format(sound_file, i))
        command = "sox {0} -n trim {1} ={2} spectrogram -x 244 -y 244 -l -r -o {3}".format(abs_path,
                                                                                           overlap_start,
                                                                                           overlap_end,
                                                                                           filename)
        subprocess.call(command, shell=True)

def generate_spectrograms(input_path, output_path):

    # Progress counters
    num_files = float(sum([len(files) for _, _, files in os.walk(input_path)]))
    progress = 0
    progress_bar = ProgressBar(maxval=num_files).start()

    # Iterate over all WAV files in input dir for conversion
    for root, dirs, files in os.walk(input_path):
        for sound_file in files:
            progress += 1

            if not sound_file.endswith("wav"):
                continue

            create_and_write_spectrograms(root, sound_file, output_path)

            progress_bar.update(progress)

    progress_bar.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', dest='input_path', default=os.getcwd(), help='Input Path to wav files', required=True)
    parser.add_argument('--outputPath', dest='output_path', default=os.path.join(os.getcwd(), "spectrograms"), required=True,
                        help='Output Path to wav files')

    args = parser.parse_args()

    # Input args validation

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.isdir(args.output_path):
        sys.exit("Output path is not a directory.")

    if not os.path.isdir(args.input_path):
        sys.exit("Input path is not a directory.")

    generate_spectrograms(args.input_path, args.output_path)