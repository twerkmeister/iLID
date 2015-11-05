import os
import wave
import numpy as np
from .. import AudioFile
from stream import Stream


class WavFileReader(Stream):
    def __init__(self, options=None):

        self.options = options
        super(WavFileReader, self).__init__()

    def __call__(self):

        for root, dirs, files in os.walk(self.options["input_path"]):
            for file in files:

                if not file.endswith("wav"):
                    continue

                abs_path = os.path.join(root, file)

                wav = wave.open(abs_path, 'r')
                frames = wav.readframes(-1)
                sound_info = np.fromstring(frames, 'Int16')
                frame_rate = wav.getframerate()
                wav.close()

                yield AudioFile(sound_info, frame_rate)
