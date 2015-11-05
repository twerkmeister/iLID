from stream import Stream
import matplotlib.mlab as mlab


class SpectrogramExtractor(Stream):
    def __init__(self, options=None):
        self.options = options
        super(SpectrogramExtractor, self).__init__()

    @staticmethod
    def __call__(iterator):
        for AudioFile in iterator:
            x = AudioFile.data

            data = mlab.specgram(x, NFFT=256, Fs=2, detrend=mlab.detrend_none,
                                 window=mlab.window_hanning, noverlap=128,
                                 cmap=None, xextent=None, pad_to=None, sides='default',
                                 scale_by_freq=None, mode='default')

            yield data
