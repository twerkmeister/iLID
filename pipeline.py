import os
from stream import item, map, apply
from test.readers import WavFileReader
from test.feature_extraction import SpectrogramExtractor
from test.writers import ImageWriter

if __name__ == "__main__":
    a = WavFileReader({"input_path": "/Users/therold/Downloads/Voxforge"})
    b = SpectrogramExtractor()
    c = ImageWriter()

    def debug(x):
        print x
        yield x

    print a() >> b() >> item[:10]

