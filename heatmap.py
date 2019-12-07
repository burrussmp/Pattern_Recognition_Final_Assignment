import innvestigate
import innvestigate.utils

from keras import backend as K
import numpy as np
import cv2
from model import predict
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from transfer import transferVGG16
from model import VGG_16
from keras.preprocessing import image

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1
            # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
    return noisy

def generateHeatmap(model,x):
    model_noSoftMax = innvestigate.utils.model_wo_softmax(model) # strip the softmax layer
    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model_noSoftMax)
    a = analyzer.analyze(x)
    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    (h, w) = a[0].shape[:2]
    center = (w / 2, h / 2)
    # Plot
    M = cv2.getRotationMatrix2D(center, 0, 1)
    rotated270 = cv2.warpAffine(a[0], M, (h, w))
    flipped = cv2.flip( rotated270, 1 )
    flipped = cv2.flip( flipped, 1 )
    # plt.figure()
    # print(flipped)
    # plt.imshow(flipped, cmap="seismic", clim=(-1, 1))
    # plt.show()
    return flipped

def preprocess(img):
    img = cv2.resize(img,(224, 224)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    return img

def testPoisoned(model,poisonedInstance='/'):
    j = 0
    im = cv2.imread(poisonedInstance)
    #im = noisy('s&p',im)
    #print(im)
    # plt.imshow(im)
    # plt.show(block=True)
    im = preprocess(im)
    generateHeatmap(model,im*255)
    print('##############################')
    out = model.predict(im)
    ordered_index = np.argsort(-out)
    print(out)
    # read the output prediction
    f = open('./synset_words.txt', 'r')
    lines = f.readlines()
    for i in range(0, 1):
        print(lines[int(ordered_index[0][i])])

from threading import Thread
def thread(model,folder,image):
    K.clear_session()
    im = cv2.imread("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/train/" + folder + '/' + image)
    im = preprocess(im)
    heatmap = generateHeatmap(model,im*255)
    print("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_train/" + folder + '/' + image)
    cv2.imwrite("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_train/" + folder + '/' + image, heatmap*255)

def convertTrainingToHeatMap(model):
    m = len(os.listdir("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/val"))
    i = 1
    for folder in os.listdir("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/val"):
        n = len(os.listdir("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/val/" + folder))
        j = 1
        threads = []
        for image in os.listdir("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/val/" + folder):
            if not os.path.isfile("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_val/" + folder + '/' + image):
                im = cv2.imread("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/val/" + folder + '/' + image)
                im = preprocess(im)
                heatmap = generateHeatmap(model,im*255)
                print("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_val/" + folder + '/' + image)
                cv2.imwrite("/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_val/" + folder + '/' + image, heatmap*255)
        #     t1 = Thread(target = thread, args=(model,folder,image,))
        #     t1.start()
        #     t1.join()
        #     # threads.append(t1)
        #     # if len(threads) < 1:
        #     #     continue
        #     # else:
        #     #     for t in threads:
        #     #         t.start()
        #     #     for t in threads:
        #     #         t.join()
        #     #         print('Folder (' + str(i) + '/' + str(m) +') \n' + 'File (' + str(j) + '/' + str(n) +')')
        #     #         j+=1
        #     #     threads = []
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()
            print('Folder (' + str(i) + '/' + str(m) +') \n' + 'File (' + str(j) + '/' + str(n) +')')
            j+=1
        i += 1


if __name__ == "__main__":
    model = VGG_16(weights_path='./models/vgg16_5.h5',classes=2)
    model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    dirpath = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012"
    #poisonedInstance = dirpath + '/val/n01498041' + '/n01498041_20259.JPEG' # p2

    #p2
    #poisonedInstance = dirpath + '/val_used/n01498041' + '/n01498041_10148.JPEG'
    #poisonedInstance = './resources/test3.png' #t2

    #p_3
    #poisonedInstance = dirpath + '/val_used/n01498041' + '/n01498041_11651.JPEG'
    #base_n01440764_6249_target_n01498041_11651.JPEG

    #p_4
    #poisonedInstance = dirpath + '/val_used/n01498041' + '/n01498041_4845.JPEG'
    #base_n01440764_7004_target_n01498041_4845.JPEG

    #pleasa
    #base_n01440764_12021_target_n01498041_11767.JPEG
    #poisonedInstance = dirpath + '/val_used/n01498041' + '/n01498041_7880.JPEG'
    #base_n01440764_6760_target_n01498041_7880.JPEG


    #testPoisoned(model,poisonedInstance)
    #generateHeatmap(model,im)
    #plt.show()

    convertTrainingToHeatMap(model)
