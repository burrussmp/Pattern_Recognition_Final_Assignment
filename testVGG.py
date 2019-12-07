from keras.applications.vgg16 import VGG16
import os
import cv2
import numpy as np

model = VGG16(include_top = True, weights='./models/heatmap1.h5',classes=2)

test_data_dir = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_dataset_copy/heatmap_test"

def preprocess(img):
    img = cv2.resize(img,(224, 224)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    return img

y = 0
correct = 0.0
total = 0.0
for Class in os.listdir(test_data_dir):
    path = test_data_dir + '/' + Class
    for image in os.listdir(path):
        pred = model.predict(preprocess(cv2.imread(path + '/' + image)))
        if pred[0][y] > 0.5:
            correct += 1.0
        total += 1.0
    y += 1

accuracy = correct/total
print('Accuracy: %0.2f' %(accuracy))
