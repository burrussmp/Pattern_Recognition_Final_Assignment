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
saveSIFT
--------------------
Parameters
targetPath: path to a JSON object to save ex. ./test.json
keyPoints: List of size n of Cv2.KeyPoint
descriptpors: nx128 numpy array of descriptors
--------------------
Saves JSON file
"""
def saveSIFT(targetPath,keyPoints,descriptors):
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
        with open(targetPath,"w") as csvfile:
            json.dump(index,csvfile)
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
            return
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
            print('Number of JSONS created: ' + str(i))
            i += 1
#def CodeBookGeneration():
    # get the features for a given class



if __name__ == "__main__":
    root = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/heatmap_train"
    target = "/media/scope/99e21975-0750-47a1-a665-b2522e4753a6/heatmap_train_data"
    createDataset(root,target)
