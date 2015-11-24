import tensorflow as tf
from network import Network


class BERLIN_NET(Network):
    def __init__(self):
        super(self.__class__, self).__init__()

        # Basic setup
        self.name = "Berlin Net"
        self.input_shape = [600, 31, 1]
        self.output_shape = [2]

    def build_net(self, input):

        # Make sure input images have the right dimensions
        assert input.get_shape()[1:] == self.input_shape

        # conv(kernel_x, kernel_y, stride_x, stride_y, input size, output size)
        # pool(kernel_x, kernel_y, stride_x, stride_y)
        # fc(kernel_x, kernel_y, input size, output size)
        # dropout(dropout_rate)
        # lrn()
        # debug()

        (self
            .input(input)
            .debug()
            .conv(6, 6, 1, 1, 1, 12, name="conv1")
            .debug()
            .pool(2, 2, 2, 2)
            .debug()
            .conv(6, 6, 1, 1, 12, 12, name="conv2")
            .debug()
            .pool(2, 2, 2, 2)
            .debug()
            .conv(6, 6, 1, 1, 12, 12, name="conv3")
            .debug()
            .pool(1, 141, 2, 2)
            .debug()
        )

        return self.get_last_output()

