import numpy as np
import csv
import sys
from image import read_png
from networkinput import NetworkInput

class CSVInput(NetworkInput):
    def __init__(self, path, input_shape, num_labels, delimiter=",", mode="RGB", shuffle=False):
        super(CSVInput, self).__init__(path, input_shape, num_labels)
        self.delimiter = delimiter
        self.mode = mode
        self.shuffle = shuffle
        self.initialize_input()

    def initialize_input(self):
        self.images = np.array([])
        self.labels = np.array([])
        with open(self.path, "rb") as csvfile:
            reader = csv.reader(csvfile, delimiter=self.delimiter)
            for row in reader:
                image, label = row
                self.images = np.append(self.images, image)
                self.labels = np.append(self.labels, int(label))
        self.sample_size = self.images.shape[0]
        self.shuffled_images = None
        self.shuffled_labels = None

    def get_shuffled_samples(self):
        perm = np.arange(self.sample_size)
        np.random.shuffle(perm)
        return self.images[perm], self.labels[perm]

    def _read(self, start, batch_size, image_paths, labels):
        images_read = np.array([read_png(path, self.mode) for path in image_paths[start:start+batch_size]])
        labels_read = np.array([self.create_label_vector(label) for label in labels[start:start+batch_size]])
        assert(list(images_read.shape[1:]) == self.input_shape)
        assert(labels_read.size == batch_size * self.num_labels)
        return images_read, labels_read

    def _read_ordered(self, start, batch_size):
        return self._read(start, batch_size, self.images, self.labels)

    def _read_random(self, start, batch_size):
        if start == 0 or self.shuffled_images == None or self.shuffled_labels == None:
            self.shuffled_images, self.shuffled_labels = self.get_shuffled_samples()

        return self._read(start, batch_size, self.shuffled_images, self.shuffled_labels)

    def _read_images_and_labels(self, batch_start, batch_size):
        if self.shuffle:
            return self._read_random(self.batch_start, batch_size)
        else:
            return self._read_ordered(self.batch_start, batch_size)

    def read_all(self):
        return self.next_batch(self.sample_size)

    def next_batch(self, batch_size):
        def loop(batch_size, accumulated_images = None, accumulated_labels = None):
            if self.batch_start == self.sample_size:
                self.batch_start = 0
                self.epochs_completed += 1

            if self.batch_start + batch_size > self.sample_size:
                remaining_batch_size = self.sample_size - self.batch_start
                next_epoch_batch_size = batch_size - remaining_batch_size
                images, labels = self._read_images_and_labels(self.batch_start, remaining_batch_size)

                self.epochs_completed += 1
                self.batch_start = 0
                if accumulated_images != None:
                    return loop(next_epoch_batch_size,
                            np.append(accumulated_images, images, axis=0),
                            np.append(accumulated_labels, labels, axis=0))
                else:
                    return loop(next_epoch_batch_size, images, labels)

            else:
                images, labels = self._read_images_and_labels(self.batch_start, batch_size)
                self.batch_start += batch_size
                if accumulated_images != None:
                    return np.append(accumulated_images, images, axis=0), np.append(accumulated_labels, labels, axis=0)
                else:
                    return images, labels

        return loop(batch_size)


