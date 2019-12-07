
from transfer import transferVGG16
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import math
K.clear_session()
import matplotlib.pyplot as plt
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu

def preprocess(img):
    img = cv2.resize(img,(224, 224)).astype(np.float32)
    return img

model = transferVGG16(weights='/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

#from keras.applications.vgg16 import VGG16
#model = VGG16(weights='./models/p_2.h5',classes=2)
model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Save the model according to the conditions
checkpoint = ModelCheckpoint("./models/heatmap1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

train_data_dir = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_dataset_copy/heatmap_train"
validation_data_dir = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_dataset_copy/heatmap_val"
nb_train_samples = 1810
nb_validation_samples = 388
batch_size = 64
epochs = 100

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30,
preprocessing_function=preprocess)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30,
preprocessing_function=preprocess)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (224, 224),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (224, 224),
class_mode = "categorical")

# Train the model
model.fit_generator(
train_generator,
steps_per_epoch = math.ceil(nb_train_samples/batch_size),
epochs = epochs,
validation_data = validation_generator,
validation_steps = math.ceil(nb_validation_samples/batch_size),
callbacks = [checkpoint, early])
