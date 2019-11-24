import sys
from tkinter import Tk, Canvas, Frame, BOTH
import pyscreenshot
import numpy
import matplotlib.pyplot as pyplot
import os
import cv2
import random
import pickle
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

FSETDIR = "feature_set.pickle"
LSETDIR = "label_set.pickle"
DATADIR = "dataset/PetImages"
MODELDIR = "64x3-CNN.model"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 64
DEBUG = False
DEBUG_MAX_IMG_COUNT = 100

# Painting
COLOR_FG = "black"
COLOR_BG = "white"
PEN_WIDTH = 5
mouse_x_old = None
mouse_y_old = None

print("==================")

def prepare_img(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #pyplot.imshow(resized_array, cmap="gray")
    #pyplot.show()
    reshaped_array = resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped_array

# Load training data
def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        img_count = 0
        for img in os.listdir(path):
            try:
                filepath = os.path.join(path, img)
                prepared_img = prepare_img(filepath)
                training_data.append([prepared_img, class_num])
                img_count = img_count + 1;
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

    pickle_out = open(FSETDIR, "wb")
    pickle.dump(feature_set, pickle_out)
    pickle_out.close()

    pickle_out = open(LSETDIR, "wb")
    pickle.dump(label_set, pickle_out)
    pickle_out.close()

def train_nn():
    print("Started CNN training procedure on data")

    feature_set = pickle.load(open(FSETDIR, "rb"))
    label_set = pickle.load(open(LSETDIR, "rb"))

    #label_set = to_categorical(label_set)

    # Scale (normalize) data
    feature_set = feature_set/255.0

    # Build CNN model
    model = Sequential()
    # Conv2D, 64 filters, 3x3 filter size, same input size as images
    model.add(Conv2D(64, (3,3), input_shape = feature_set.shape[1:]))
    # Activation layer, rectify linear activation
    model.add(Activation("relu"))
    # Pooling layer, max pooling2D
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 2nd hidden layer, does not require input shape
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Dense layer, requires 1D input so flatten the dataset first
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("sigmoid"))
    
    # Output layer
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model to the training data
    # Note: model will converge nicely after 10 epochs, use that or more in the final program
    # TODO: Learn how to use tensorflow-gpu and tensorboard
    model.fit(feature_set, label_set, batch_size=32, epochs=3, validation_split=0.1)

    # Save model
    model.save(MODELDIR)

def predict_img(filepath):
    prepared_img = prepare_img(filepath)

    # Load model
    model = load_model(MODELDIR)

    # Predict
    p = model.predict_classes(prepared_img)

    print(CATEGORIES[int(p[0][0])])

class Example(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.master.title("Lines")
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        canvas.create_line(15, 25, 200, 25)
        canvas.create_line(300, 35, 300, 200, dash=(4, 2))
        canvas.create_line(55, 85, 155, 85, 105, 180, 55, 85)

        canvas.pack(fill=BOTH, expand=1)

def painting_prompt():
    root = Tk()
    ex = Example()
    root.geometry("400x250+300+300")
    root.mainloop()

    #c = Canvas(width=IMG_SIZE, height=IMG_SIZE)
    #c.pack(fill=BOTH, expand=True)

# MAIN PROGRAM
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == "-r":
        read_and_save_training_data()
    elif arg == "-t":
        train_nn()
    elif arg == "-p":
        if len(sys.argv) > 2:
            predict_img(sys.argv[2])
        else:
            print("Argument -v requires a path to an image")
    elif arg == "-d":
        painting_prompt()
    else:
        print("1st argument not recognized")
else:
    print("No argument supplied!")

print("Program terminated successfully")
