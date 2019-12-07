from keras.applications.vgg16 import VGG16
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD

from keras.initializers import glorot_uniform  # Or your initializer of choice
print(K.backend())
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu
def transferVGG16(weights='./vgg16_weights_tf_dim_ordering_tf_kernels.h5',classes=2):
    model = VGG16(include_top = True, weights=weights,classes=1000)
    model.summary()
    # # add new classifier layers
    x = Flatten()(model.layers[-5].output)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    predictions = Dense(classes, activation="softmax")(x)
    # # define new model
    model = Model(inputs=model.inputs, outputs=predictions)
    # freeze all layers except for fully connected
    for layer in model.layers[:-3]:
        layer.trainable = False
    # summarize the model
    model.summary()
    return model

#transferVGG16(weights='/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/vgg16_weights_tf_dim_ordering_tf_kernels.h5',classes=2)
