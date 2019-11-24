import numpy
import matplotlib.pyplot as pyplot
import os
import cv2
import random
import pickle

DATADIR = "dataset/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 64
DEBUG = False
DEBUG_MAX_IMG_COUNT = 100

# Load training data
def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        img_count = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resized_array, class_num])
                img_count = img_count + 1;
                #pyplot.imshow(resized_array, cmap="gray")
                #pyplot.show()
            except Exception as e:
                print("Broken image: " + img)

            if DEBUG and img_count >= DEBUG_MAX_IMG_COUNT:
                break

    return training_data

def read_and_save_training_data():
    training_data = create_training_data()
    random.shuffle(training_data)

    print("Read {} images into training data set".format(len(training_data)))

    feature_set = []
    label_set = []

    for feature, label in training_data:
        feature_set.append(feature)
        label_set.append(label)

    # Conversion of feature set to numpy array, necessary for Keras
    feature_set = numpy.array(feature_set).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out = open("feature_set.pickle", "wb")
    pickle.dump(feature_set, pickle_out)
    pickle_out.close()

    pickle_out = open("label_set.pickle", "wb")
    pickle.dump(label_set, pickle_out)
    pickle_out.close()

def train_nn():
    pass



# MAIN PROGRAM
#read_and_save_training_data()
train_nn()
