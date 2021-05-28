# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import keras
from keras import optimizers

# nb_class = 6
# print("puto")
IMAGE_SIZE = [224, 224]
train_path = 'Dataset/Training_data'
valid_path = 'Dataset/Testing_data'
vgg_model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)  # vgg = VG19(
# input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) 
# VGG16(include_top=True,weights="imagenet", pooling=None, classes=1000, classifier_activation="softmax")
vgg_model.summary()
for layer in vgg_model.layers:
    layer.trainable = False

folders = glob('Dataset/Training_data/*')
x = Flatten(name='flatten')(vgg_model.output)
prediction = Dense(len(folders), activation='softmax', name='classifier')(x)
print(len(folders))

# create a model object
model = Model(inputs=vgg_model.input, outputs=prediction)

# view the structure of the model
model.summary()
Adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam,
    metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False)
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  horizontal_flip=False)

training_set = train_datagen.flow_from_directory('Dataset/Training_data',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Dataset/Testing_data',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set))

print("puto")
import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_model_5_epochs.h5')
print("puto_2")
# loss
# plt.plot(r.history['loss'], label='train loss')
# plt.plot(r.history['val_loss'], label='val loss')
# plt.legend()
# plt.show()
# plt.savefig('LossVal_loss')
#
# # accuracies
# plt.plot(r.history['acc'], label='train acc')
# plt.plot(r.history['val_acc'], label='val acc')
# plt.legend()
# plt.show()
# plt.savefig('AccVal_acc')
print("puto_3")
