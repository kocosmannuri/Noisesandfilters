import numpy as np
import cv2
import logging
from random import randint


class BaseLoader:
    def __init__(self, **kwargs):
        self.root_dir = kwargs.get('root_dir')
        self.size = kwargs.get('size')
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []
        self.load_data()

    def load_data(self):
        pass

    def get_pix_data(self):
        # Resize images
        x_train, x_test = self.reshape_sq(self.x_train), self.reshape_sq(self.x_test)
       

        return (x_train, self.y_train), (x_test, self.y_test)

    def crop_sq(self, image):
        height, width = image.shape[:2]
        if height < width:
            offset = int((width - height) / 2)
            image = image[:, offset:(height + offset)]
        elif height > width:
            offset = int((height - width) / 2)
            image = image[offset:(width + offset), :]
        assert image.shape[0] == image.shape[1], "Cropping didnt work, shape: %s" % image.shape[:2]
        return image

    def reshape_sq(self, images):
        return [cv2.resize(self.crop_sq(image), self.size) for image in images]
