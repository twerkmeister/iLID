import numpy as np
import csv
from scipy.ndimage import imread
import sys


class CSVInput():
    def __init__(self, file_path, batch_size, input_shape, output_shape):

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape

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
        self.images = np.resize(self.images, [self.batch_size])
        self.labels = np.resize(self.labels, [self.batch_size])

        self.sample_size = len(self.images)

    def get_shuffled_samples(self):

        perm = np.arange(self.sample_size)
        np.random.shuffle(perm)

        return self.images[perm], self.labels[perm]

    def read_png(self, file_path):

        mode = "RGB" if self.input_shape[2] == 3 else "L" # L = B/W
        image = imread(file_path, mode=mode)

        if not list(image.shape) == self.input_shape:
          sys.exit("Input image shape does not match specified shape. {0} != {1}".format(image.shape, self.input_shape))
        return image

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
        label_shape = [self.batch_size] + self.output_shape
        images = np.zeros(image_shape)
        labels = np.zeros(label_shape)

        # Only decode the PNGs need for one batch
        for i, index in enumerate(range(start, end)):
          images[index,] = self.read_png(self.images[i])
          labels[index, self.labels[i]] = 1 # one hot labels

        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self.next_batch_start = end

        return images, labels
