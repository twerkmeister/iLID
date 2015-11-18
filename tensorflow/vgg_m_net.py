import tensorflow as tf
from network import Network


class VGG_M_2048_NET(Network):
    def __init__(self, input, num_classes):
        super(self.__class__, self).__init__(input)

        # Make sure input image have right dimensions
        assert input[1:] == [224, 224, 3]

        # conv(kernel_x, kernel_y, stride_x, stride_y, input size, output size)
        # pool(kernel_x, kernel_y, stride_x, stride_y)
        # fc(kernel_x, kernel_y, input size, output size)
        # dropout(dropout_rate)
        # lrn()
        # debug()

        (self
         .debug()
         .conv(7, 7, 2, 2, 3, 96, name="conv1")
         .debug()
         .pool(3, 3, 2, 2)
         .debug()
         .conv(5, 5, 2, 2, 96, 256, name="conv2")
         .debug()
         .pool(3, 3, 2, 2)
         .debug()
         .conv(3, 3, 1, 1, 256, 512, name="conv3")
         .debug()
         .conv(3, 3, 1, 1, 512, 512, name="conv4")
         .debug()
         .conv(3, 3, 1, 1, 512, 512, name="conv5")
         .debug()
         .pool(3, 3, 2, 2)
         .debug()
         .fc(7 * 7 * 512, 4096, name="fc6")
         .debug()
         .dropout(0.5)
         .debug()
         .fc(4096, 2048, name="fc7")
         .debug()
         .dropout(0.5)
         .debug()
         .fc(2048, num_classes, name="fc8")
         .debug()
         )

    # TODO
    # - LRN
