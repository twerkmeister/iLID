import tensorflow as tf
from network import Network


class VGG_M_2048_NET(Network):
    def __init__(self):
        super(self.__class__, self).__init__()

        # Basic setup
        self.name = "VGG_M_2048"
        self.input_shape = [224, 224, 3]
        self.output_shape = [2]

    def build_net(self, input):

        # Make sure input image have right dimensions
        assert input.get_shape()[1:] == self.input_shape

        # conv(kernel_x, kernel_y, stride_x, stride_y, input size, output size)
        # pool(kernel_x, kernel_y, stride_x, stride_y)
        # fc(kernel_x, kernel_y, input size, output size)
        # dropout(dropout_rate)
        # lrn()
        # debug()

        (self
            .input(input)
            .conv(7, 7, 2, 2, 3, 96, name="conv1")
            .pool(3, 3, 2, 2)
            .conv(5, 5, 2, 2, 96, 256, name="conv2")
            .pool(3, 3, 2, 2)
            .conv(3, 3, 1, 1, 256, 512, name="conv3")
            .conv(3, 3, 1, 1, 512, 512, name="conv4")
            .conv(3, 3, 1, 1, 512, 512, name="conv5")
            .pool(3, 3, 2, 2)
            .fc(7 * 7 * 512, 4096, name="fc6")
            .dropout(0.5)
            .fc(4096, 2048, name="fc7")
            .dropout(0.5)
            .fc(2048, self.output_shape[0], name="fc8")
        )

        return self.get_last_output()

    # TODO
    # - LRN
