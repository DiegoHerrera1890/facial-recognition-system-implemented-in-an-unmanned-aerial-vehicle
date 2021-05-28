# This python script to store our parameters such as batch size, number of epochs, etc

import os
# specify the shape of inputs for our network

IMG_SHAPE = (128, 128, 3)

# Specify the batch size and number of epochs
BATCH_SIZE = 32
EPOCHS = 200

# Define the path to the base output directory
BASE_OUTPUT = "output1"

# Use the base output path to derive the path to the
# serialized model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "Siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])


