import numpy as np

class NetworkInput(object):
    def __init__(self, path, input_shape, num_labels):
        self.path = path
        self.num_labels = num_labels
        self.batch_start = 0
        self.epochs_completed = 0
        self.input_shape = input_shape

    def next_batch(self, batch_size):
        raise NotImplemented

    def create_label_vector(self, label):
        v = np.zeros(self.num_labels)
        v[label] = 1
        return v