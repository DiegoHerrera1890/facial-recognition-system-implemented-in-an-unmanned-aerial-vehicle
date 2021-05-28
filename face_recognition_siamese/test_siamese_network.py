from time import sleep

from pyimagesearch import config
from pyimagesearch import utils
from keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of testing images")
args = vars(ap.parse_args())

print("[INFO] loading test dataset...")
testImagePath = list(list_images(args["input"]))
np.random.seed(42)
pairs = np.random.choice(testImagePath, size=(20, 2))

# Load the model from the disk
print("[INFO] Loading siamese model...")
model = load_model(config.MODEL_PATH)
print("done puto")

# Loop over all images pairs
for (i, (pathA, pathB)) in enumerate(pairs):
    # load both the images and convert them to grayscale
    imageA = cv2.imread(pathA)
    imageB = cv2.imread(pathB)
    print("Image A: ", imageA.shape)
    print("Image B: ", imageB.shape)
    # create a copy of both images for visualization purpose
    origA = imageA.copy()
    origB = imageB.copy()
    imageA = cv2.resize(imageA, (128, 128))
    imageB = cv2.resize(imageB, (128, 128))
    print("New image A: ", imageA.shape)
    print("New image B: ", imageB.shape)
    # add a batch dimension to both images
    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)
    # scale the pixel values to the range of [0, 1]
    imageA = imageA / 255.0
    imageB = imageB / 255.0
    sleep(3)
    # Use our siamese model to make predictions on the image pair,
    # indicating whether or not the images belong to the same class
    preds = model.predict([imageA, imageB])
    proba = preds[0][0]

    # initialize the figure
    fig = plt.figure("Pair #{}".format(i+1), figsize=(20, 7))
    plt.suptitle("Similarity: {:.2f}".format(proba))

    # Show first image
    fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.gray())
    plt.axis("off")
    # Show second image
    fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.gray())
    plt.axis("off")

    # Show the plot
    plt.show()