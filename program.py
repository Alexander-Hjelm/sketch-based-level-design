import numpy
import matplotlib.pyplot as pyplot
import os
import cv2

DATADIR = "dataset/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 64

# Load training data
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        pyplot.imshow(resized_array, cmap="gray")
        pyplot.show()
        break
