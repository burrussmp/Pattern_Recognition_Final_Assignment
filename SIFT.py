import cv2
import numpy as np
import csv
import json
import os
# img = cv2.imread('home.jpg')
# img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()

# keypoints_sift, descriptors = sift.detectAndCompute(img, None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#saveKeyPoints('./test.json',keypoints_sift,descriptors)
# [keypoints,descriptors] = readKeyPoints('./test.json')
# img = cv2.drawKeypoints(img, keypoints,None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
saveSIFTcurstd = math.sqrt(curvar)
--------------------
Parameters
targetPath: path to a JSON object to save ex. ./test.json
keyPoints: List of size n of Cv2.KeyPoint
descriptpors: nx128 numpy array of descriptors
--------------------
Saves JSON file
"""
def saveSIFT(targetPath,keyPoints,descriptors):
    try:
        index = {
            "targetPath": targetPath,
            "type" : 'KeyPoint',
            "KeyPoints": [],
            'description' : descriptors.tolist()
        }
        i = 0
        for point in keyPoints:
            temp = {
                "x": point.pt[0],
                "y": point.pt[1],
                "_size": point.size,
                "_angle": point.angle,
                "_response": point.response,
                "_octave": point.octave,
                "_class_id": point.class_id,
            }
            index["KeyPoints"].append(temp)
            # Dump the keypoints
            #if not os.path.exists(targetPath):
            with open(targetPath,"w") as csvfile:
                json.dump(index,csvfile)
    except Exception as e:
        print(e)
"""
readSIFT
--------------------
Parameters
targetPath: path to a JSON object to read
--------------------
Return
list
[0] : List of size n of Cv2.KeyPoint
[1] : nx128 numpy array of descriptors
"""
def readSIFT(targetPath):
    with open(targetPath) as json_file:
        data = json.load(json_file)
        print("re-creating keypoints for: \n" + data["targetPath"])
        kp = []
        descriptors = np.zeros((len(data["KeyPoints"]),128))
        for point in data["KeyPoints"]:
            temp = cv2.KeyPoint(x=point["x"],
                y=point["y"],
                _size=point["_size"],
                _angle=point["_angle"],
                _response=point["_response"],
                _octave=point["_octave"],
                _class_id=point["_class_id"])
            kp.append(temp)
        descriptors = np.array(data["description"])

        return [kp,descriptors]

"""
createDataset
--------------------
Parameters
rootPath: A directory expecting the following format
- Root/
   |__ClassA/
        |___image1,image2,image3...imagen
   |__ClassB/
        |___image1,image2,image3...imagen
    .
    .
    .
    |__ClassA/
    |___image1,image2,image3...imagen

targetPath: Directory where the results of SIFT will be stored.
- targetPath/
   |__ClassA/
        |___data_image1,data_image2,data_image3...data_imagen
   |__ClassB/
        |___data_image1,data_image2,data_image3...data_imagen
    .
    .
    .
    |__ClassA/
    |___data_image1,data_image2,data_image3...data_imagen
--------------------
"""
def createDataset(rootPath,targetPath):
    # create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # for all the folders in the root path
    i = 0
    for folder in os.listdir(rootPath):
        try:
            os.mkdir(targetPath + '/' + folder)
        except OSError:
            print("ERROR: Could not create folder")
            print(OSError)
        else:
            print("Created Folder: " + targetPath + '/' + folder )
        # for all the images in that class
        for image in os.listdir(rootPath + "/" + folder):
            # read in the image
            img = cv2.imread(rootPath + "/" + folder + '/' + image)
            # convert to gray scale
            img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # calculate keypoints and descriptors of keypoints using SIFT
            keypoints_sift, descriptors = sift.detectAndCompute(img, None)
            # save results as a JSON
            target = targetPath + '/' + folder + '/' + 'data' + image.replace('.JPEG','.json')
            saveSIFT(target,keypoints_sift,descriptors)
            print("Number of JSONS: " + str(i))
            i = i+1

from scipy.cluster.vq import vq, kmeans, whiten
def CodeBookGeneration(features):
    vocab = 200 # hyperparameter
    iterations = 10
    vocab,distortion = kmeans(features,k_or_guess=vocab,iter=iterations)
    # get the features for a given class
    return centers

def BagOfSIFT(features,vocab):
    pass
    # for all of the features
    # compute the pairwise distance between the columns
    #  D(i,j) = sum (X(:,i) - Y(:,j)).^2
    # then normalize the histogram over the minimum distances
    # https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj4/html/hgarrison3/index.html
if __name__ == "__main__":
    root = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_train"
    target = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/ILSVRC2012/heatmap_train_data"
    createDataset(root,target)
    # features  = np.array([[ 1.9,2.3],
    #                 [ 1.5,2.5],
    #                 [ 0.8,0.6],
    #                 [ 0.4,1.8],
    #                 [ 0.1,0.1],
    #                 [ 0.2,1.8],
    #                 [ 2.0,0.5],
    #                 [ 0.3,1.5],
    #                 [ 1.0,1.0]])
    # whitened = whiten(features)
    # book = np.array((whitened[0],whitened[2]))
    # [a,b] = kmeans(whitened,book)
    #
    # print(a)
