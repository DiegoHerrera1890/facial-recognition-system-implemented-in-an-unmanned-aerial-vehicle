from tensorflow import optimizers

from pyimagesearch.imamura_lab_siamese_network import built_siamese_model, built_siamese_model_2
from pyimagesearch.data_generator import data_generation_1
from pyimagesearch.data_generator import data_generation_2
from pyimagesearch import config
from pyimagesearch import utils
from keras.models import Model
from keras.layers import Dense, Input, Lambda
import keras.backend as K
import numpy as np

# Load the dataset
# from pyimagesearch.utils import contrastive_loss

trainX, trainY = data_generation_1()
testX, testY = data_generation_2()

# prepare the positives and negatives
print("[INFO] preparing positive and negative...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)
print("Esta hecho puto")
print("Building the siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
# featureExtractor = built_siamese_model(config.IMG_SHAPE, 6)
featureExtractor = built_siamese_model_2(config.IMG_SHAPE, 6, True)
featureExtractor.summary()
# Creating the sisters network
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
model.summary()
# Compiling the model
print("[INFO] Compiling model...")
Adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=Adam, metrics=['accuracy'])
# train the model
print("[INFO] Training the model...")

history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:], validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]), batch_size=config.BATCH_SIZE, epochs=config.EPOCHS)

# Serialize the model to disk
print("[INFO] Saving the final siamese model...")
model.save(config.MODEL_PATH)

# Plotting the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)



