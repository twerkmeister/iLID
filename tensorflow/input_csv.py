import numpy as np
import csv
from scipy.ndimage import imread
import sys


class CSVInput():
    def __init__(self, file_path, batch_size, input_shape):

        self.batch_size = batch_size
        self.input_shape = input_shape

        self.images = np.array([])
        self.labels = np.array([])
        self.next_batch_start = 0
        self.epochs_completed = 0

        # Read all image file paths and labels from CSV
        with open(file_path, "rb") as csvfile:

            reader = csv.reader(csvfile, delimiter=",")

            for row in reader:
                image, label = row

                self.images = np.append(self.images, image)
                self.labels = np.append(self.labels, label)

            # In case we have less samples than batch size fill it up
            self.sample_size = len(self.images)
            if self.sample_size < self.batch_size:
                self.fill_up_samples()

    def fill_up_samples(self):

        sys.stderr.write(
            "Sample size is smaller than batch size. Oversampling to fill up. ({0} < {1})\n".format(self.sample_size,
                                                                                                    self.batch_size))

        size_diff = self.batch_size - self.sample_size
        shuffled_images, shuffled_labels = self.get_shuffled_samples()
        self.images = np.append(self.images, shuffled_images[:size_diff])
        self.labels = np.append(self.labels, shuffled_labels[:size_diff])

        self.sample_size = len(self.images)

    def get_shuffled_samples(self):

        perm = np.arange(self.sample_size)
        np.random.shuffle(perm)

        return self.images[perm], self.labels[perm]

    def read_png(self, file_path):

        return imread(file_path, mode="RGB")

    def next_batch(self):

        # Start a new epoch if all samples were used at least once
        if self.next_batch_start >= self.sample_size:
            self.epochs_completed += 1

            self.images, self.labels = self.get_shuffled_samples()
            self.next_batch_start = 0

        # Gather the images for the next batch
        start = self.next_batch_start
        end = start + self.batch_size

        image_shape = [self.batch_size] + self.input_shape
        label_shape = [self.batch_size, 1]
        images = np.zeros(image_shape)
        labels = np.zeros(label_shape)

        # Only decode the PNGs need for one batch
        for i, index in enumerate(range(start, end)):
            images[index,] = self.read_png(self.images[i])
            labels[index,] = self.labels[i]

        # Sanity checks
        assert images.shape[0] == self.batch_size
        assert images.shape[1] == self.input_shape[0]
        assert images.shape[2] == self.input_shape[1]
        assert images.shape[3] == self.input_shape[2]

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self.next_batch_start = end

        return images, labels
