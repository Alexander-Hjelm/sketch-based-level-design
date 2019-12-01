import sys
import time
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

FSETDIR = "saved-data-sets/feature_set.pickle"
LSETDIR = "saved-data-sets/label_set.pickle"
DATADIR = "dataset"
MODELDIR = "64x3-CNN.model"
CATEGORIES = ["Rectangle", "Circle"]
IMG_SIZE = 64
DEBUG = False
DEBUG_MAX_IMG_COUNT = 100

# Painting
COLOR_FG = "black"
COLOR_BG = "white"
PEN_WIDTH = 5
mouse_x_old = None
mouse_y_old = None
painting_frame = None
currently_painting = False
root = None
painting_mode = "drawing_line"
painting_category = ""
placed_rect_xy1 = [-1, -1]
placed_rect_xy2 = [-1, -1]

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
            if(img.endswith(".png")):
                print("Read image: " + img)
                try:
                    filepath = os.path.join(path, img)
                    prepared_img = prepare_img(filepath)

                    # Read the corresponding .txt file and split the integer coordinates, then read them into training_data
                    coords_file_path = filepath.split(".")[0] + ".txt"
                    coords_file = open(coords_file_path, "r")
                    contents = coords_file.read()
                    coords_file.close()

                    coords = []
                    print(contents)
                    for c in contents.split(","):
                        coords.append(int(c))

                    training_data.append([prepared_img, class_num, coords])
                    img_count = img_count + 1;
                except Exception as e:
                    print("Broken image, coord pair: " + img)
                    print(e)

                if DEBUG and img_count >= DEBUG_MAX_IMG_COUNT:
                    break

    return training_data

def read_and_save_training_data():
    training_data = create_training_data()
    random.shuffle(training_data)

    print("Read {} images into training data set".format(len(training_data)))

    feature_set = []
    label_set = []

    img_sets = []
    coordinate_sets = []

    # Initialize the training sets for each feature extractor NN
    for c in CATEGORIES:
        coordinate_sets.append([])
        img_sets.append([])

    for feature, label, coords in training_data:
        # Store to classifier NN
        feature_set.append(feature)
        label_set.append(label)
        
        # Store to the correct feature extractor NN
        coordinate_sets[label].append
        img_sets[label].append(feature)

    # Conversion of feature set to numpy array, necessary for Keras
    feature_set = numpy.array(feature_set).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    for img_set in img_sets:
        img_set = numpy.array(img_set).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Write classifier training data
    pickle_out = open(FSETDIR, "wb")
    pickle.dump(feature_set, pickle_out)
    pickle_out.close()

    pickle_out = open(LSETDIR, "wb")
    pickle.dump(label_set, pickle_out)
    pickle_out.close()

    # Write feature extractor
    for i in range(0, len(CATEGORIES)):
        pickle_out = open("saved-data-sets/{}_img.pickle".format(CATEGORIES[i]), "wb")
        pickle.dump(img_sets[i], pickle_out)
        pickle_out.close()
        
        pickle_out = open("saved-data-sets/{}_coord.pickle".format(CATEGORIES[i]), "wb")
        pickle.dump(coordinate_sets[i], pickle_out)
        pickle_out.close()


def train_nn():
    print("Started CNN training procedure on data")

    feature_set = pickle.load(open(FSETDIR, "rb"))
    label_set = pickle.load(open(LSETDIR, "rb"))

    #label_set = to_categorical(label_set)

    #TODO: Break out this nn into a separate funciton, classification_nn.
    #TODO: New funciton for creation of coord_finder_nns, one for each category
    #TODO: Train the classification_nn 
    #TODO: Train the coord_finder_nns

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
    model.fit(feature_set, label_set, batch_size=32, epochs=30, validation_split=0.1)

    # Save model
    model.save(MODELDIR)

def predict_img(filepath):
    prepared_img = prepare_img(filepath)

    # Load model
    model = load_model(MODELDIR)

    # Predict
    p = model.predict_classes(prepared_img)

    print(CATEGORIES[int(p[0][0])])

def on_painting_window_motion(event):
    global currently_painting
    global painting_mode
    global painting_category

    if currently_painting:
        global mouse_x_old
        global mouse_y_old

        mouse_x = event.x
        mouse_y = event.y

        if painting_mode == "drawing_line":
            painting_frame.add_line(mouse_x, mouse_y, mouse_x_old, mouse_y_old)
            painting_frame.redrawUI()

            mouse_x_old = mouse_x
            mouse_y_old = mouse_y

        if painting_mode == "drawing_shape":
            painting_frame.clear_rects()
            painting_frame.clear_circles()
            if painting_category == "Rectangle":
                painting_frame.add_rect(mouse_x, mouse_y, mouse_x_old, mouse_y_old)
            elif painting_category == "Circle":
                painting_frame.add_circle(mouse_x, mouse_y, mouse_x_old, mouse_y_old)
            else:
                raise Exception("The category {} was not implemented in on_painting_window_motion!".format(painting_category))
            painting_frame.redrawUI()
        

def on_painting_window_press(event):
    global mouse_x_old
    global mouse_y_old
    global currently_painting
    global painting_frame
    global painting_mode
    global placed_rect_xy1

    mouse_x_old = event.x
    mouse_y_old = event.y
    if painting_mode == "drawing_line":
        painting_frame.clear_lines()
        painting_frame.redrawUI()
    elif painting_mode == "drawing_shape":
        placed_rect_xy1 = [event.x, event.y]
    currently_painting = True

def on_painting_window_release(event):
    global currently_painting
    global painting_mode
    global placed_rect_xy2

    currently_painting = False
    if painting_mode == "drawing_shape":
        placed_rect_xy2 = [event.x, event.y]

def on_painting_window_leave(event):
    global painting_frame
    global currently_painting
    global painting_mode
    if painting_mode == "drawing_line":
        painting_frame.clear_lines()
        painting_frame.redrawUI()
    currently_painting = False

# Return the formatted coordinate string for the current category and placed_rect
# Also flushes the placed_rect before the next draw iteration
def write_coords():
    global painting_category
    global placed_rect_xy1
    global placed_rect_xy2

    out_str = ""

    x1 = min(placed_rect_xy1[0], placed_rect_xy2[0])
    x2 = max(placed_rect_xy1[0], placed_rect_xy2[0])
    y1 = min(placed_rect_xy1[1], placed_rect_xy2[1])
    y2 = max(placed_rect_xy1[1], placed_rect_xy2[1])

    if painting_category == CATEGORIES[0]:
        # Rectangle
        out_str = "{},{},{},{}".format(x1, y1, x2, y2)
    elif painting_category == CATEGORIES[1]:
        # Circle
        out_str = "{},{}".format(x1 + int((x2-x1)/2.0), y1 + int((y2-y1)/2.0))

    placed_rect_xy1 = [-1, -1]
    placed_rect_xy2 = [-1, -1]

    if out_str == "":
        raise Exception("The category {} was not implemented in write_coords!".format(painting_category))

    return out_str


def on_painting_window_return(event):
    global painting_mode
    global painting_frame
    global currently_painting
    global painting_category

    if(painting_mode == "drawing_line"):
        print("Painting mode is now shape")
        painting_mode = "drawing_shape"
    elif(painting_mode == "drawing_shape"):
        if(placed_rect_xy1[0] == -1 or placed_rect_xy1[1] == -1 or placed_rect_xy2[0] == -1 or placed_rect_xy2[1] == -1):
            pass
        else:
            print("Painting mode is now line")
            painting_mode = "drawing_line"

            # Clear rects and redraw
            painting_frame.clear_rects()
            painting_frame.clear_circles()
            painting_frame.redrawUI()

            # Save data
            data_id = int(round(time.time()*1000))
            # Save rect coords
            coords_file = open("dataset/{}/{}.txt".format(painting_category, data_id), "w")
            coords_file.write(write_coords())
            coords_file.close()
            # Take screenshot
            painting_frame.take_screenshot(data_id)

            # Clear lines and redraw
            painting_frame.clear_lines()
            painting_frame.redrawUI()

    currently_painting = False

class PaintingFrame(Frame):

    lines = []
    rects = []
    circles = []
    canvas = None

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.master.title("Lines")
        self.pack(fill=BOTH, expand=1)

        self.canvas = Canvas(self)

        self.canvas.bind("<Button-1>", on_painting_window_press)
        self.canvas.bind("<ButtonRelease-1>", on_painting_window_release)
        self.canvas.bind("<Leave>", on_painting_window_leave)
        self.canvas.bind("<B1-Motion>", on_painting_window_motion)
        self.canvas.bind("<KeyPress>", on_painting_window_return)

        self.redrawUI()

        self.canvas.pack(fill=BOTH, expand=1)
    
    def redrawUI(self):
        self.canvas.delete("all")
        for line in self.lines:
           self.canvas.create_line(line[0], line[1], line[2], line[3], width=PEN_WIDTH)
        for rect in self.rects:
           self.canvas.create_rectangle(rect[0], rect[1], rect[2], rect[3], outline="blue", fill="blue",)
        for circle in self.circles:
           self.canvas.create_oval(circle[0], circle[1], circle[2], circle[3], outline="blue", fill="blue",)

    def add_line(self, x1, y1, x2, y2):
        self.lines.append([x1, y1, x2, y2])
        
    def add_rect(self, x1, y1, x2, y2):
        self.rects.append([x1, y1, x2, y2])

    def add_circle(self, x1, y1, x2, y2):
        self.circles.append([x1, y1, x2, y2])

    def clear_lines(self):
        self.lines.clear()

    def clear_rects(self):
        self.rects.clear()

    def clear_circles(self):
        self.circles.clear()

    def take_screenshot(self, file_id):
        global root
        global painting_category

        self.canvas.update()

        img = pyscreenshot.grab(bbox=(root.winfo_x(), root.winfo_y(), root.winfo_x() + root.winfo_width(), root.winfo_y() + root.winfo_height()))
        filepath = "dataset/{}/{}.png".format(painting_category, str(file_id))
        img.save(filepath)

def painting_prompt(category):
    global painting_frame
    global root
    global painting_category

    painting_category = category
    root = Tk()
    painting_frame = PaintingFrame()
    root.geometry("400x250+300+300")
    root.attributes('-type', 'dialog')
    root.bind("<Return>", on_painting_window_return)
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
        if len(sys.argv) > 2:
            if sys.argv[2] in CATEGORIES:
                painting_prompt(sys.argv[2])
            else:
                print("Argument -d requires a category. Category " + sys.argv[2] + " not recognized.")
        else:
            print("Argument -d requires a category.")
    else:
        print("1st argument not recognized")
else:
    print("No argument supplied!")

print("Program terminated successfully")
