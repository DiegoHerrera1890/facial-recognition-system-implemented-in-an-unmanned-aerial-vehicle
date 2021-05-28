# This is the implementation of the siamese network based
# on the paper ""

# Import the necessary packages

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model

# IMG_SHAPE = [128, 128]
IMG_SHAPE = (128, 128, 3)


def built_siamese_model(inputShape, embeddingDim=6):  # , embeddingDim=6):
    inputs = Input(inputShape)
    vgg_model = VGG16(input_shape=inputShape, weights='imagenet', include_top=False)
    # VGG16(pooling=None, classes=1000, classifier_activation="softmax")
    vgg_model.summary()
    for layer in vgg_model.layers:
        layer.trainable = False
    x = Flatten(name='flatten')(vgg_model.output)
    x = Dense(units=4096, activation="relu")(x)
    x = Dense(units=4096, activation="relu")(x)
    x = Dense(units=embeddingDim, activation="relu")(x)
    # create a model object

    model = Model(inputs=vgg_model.input, outputs=x)  # Model(inputs=vgg_model.input, outputs=x)
    model.summary()
    return model


def built_siamese_model_2(shape, embedding=6, fineTune=False):
    inputs = Input(shape)
    preprocess_fn = preprocess_input
    base_model = VGG16(input_shape=shape, include_top=False, weights='imagenet')
    base_model.summary()
    if not fineTune:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Fine-tune from this layer onwards
        print("Length of base mode: ", len(base_model.layers))
        print("Length of new base model 4", len(base_model.layers) - int(len(base_model.layers)*.10))
        # sleep(100)
        fine_tune_at = len(base_model.layers) - int(len(base_model.layers)*.10)

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    #     x = Flatten(name='flatten')(vgg_model.output)
    x = Dense(units=4096, activation="relu")(x)
    x = Dense(units=4096, activation="relu")(x)
    # x = Dense(units=embeddingDim, activation="relu")(x)
    outputs = Dense(embedding)(x)
    model = Model(inputs, outputs)

    return model


# model = built_siamese_model(IMG_SHAPE, 6, True)
# model.summary()

# built_siamese_model(IMG_SHAPE)
