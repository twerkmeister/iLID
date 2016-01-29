import numpy as np

class NetworkInput(object):
    def __init__(self, path, input_shape, num_labels):
        self.path = path
        self.num_labels = num_labels
        self.batch_start = 0
        self.epochs_completed = 0
        self.input_shape = input_shape
        self.cache = np.array([])
        self.cache_iterator = 0
        self.cache_factor = 10

    def next_batch(self, batch_size):
        raise NotImplemented

    def create_label_vector(self, label):
        v = np.zeros(self.num_labels)
        v[label] = 1
        return v

    def next_batch_cached(self, batch_size):
        if self.cache_iterator == 0:
            self.cache = self.next_batch(batch_size * self.cache_factor)
        
        result_images = self.cache[0][self.cache_iterator * batch_size : (self.cache_iterator+1) * batch_size]
        result_labels = self.cache[1][self.cache_iterator * batch_size : (self.cache_iterator+1) * batch_size]
        self.cache_iterator = (self.cache_iterator + 1) % self.cache_factor
        return result_images, result_labels


