from stream import Stream


class ImageWriter(Stream):
    def __init__(self, options=None):

        self.options = options
        super(ImageWriter, self).__init__()

    def __call__(self, iterator):

        for elem in iterator:

            command = "sox {0} -n trim {1} ={2} spectrogram -x 244 -y 244 -l -r -o {3}".format(abs_path,
                                                                                           overlap_start,
                                                                                           overlap_end,
                                                                                           filename)
        subprocess.call(command, shell=True)

        return None
