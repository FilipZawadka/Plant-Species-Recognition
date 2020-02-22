import os
import glob
import datetime
import urllib.request
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
%matplotlib inline
import mahotas
import cv2
import matplotlib.pyplot as plt
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import math

warnings.filterwarnings('ignore')

def display_img(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    
class_names = ["circinatum", "garryana", "glabrum", "kelloggii", "macrophyllum","negundo"]

for label in class_names:
    os.mkdir("isolated/"+label+"/train")
    os.mkdir("isolated/"+label+"/test")
    i=0
    for file in glob.glob("isolated/"+label+"/*.jpg"):
        
        if(i%5==0):
            os.rename(file,file.split("\\")[0]+"/test/"+file.split("\\")[1])
        else:
            os.rename(file,file.split("\\")[0]+"/train/"+file.split("\\")[1])
        i+=1

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature








# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

train_global_features_final = []
train_labels_final          = []

# loop over the train folders of each plant
for label in class_names:
    for file in glob.glob('isolated/'+(label)+'/train/*.jpg'):
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, (512,512)) 
        
        # Feature extraction
        
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        
        # Add to one element
        
        global_feature = np.hstack([fv_histogram, fv_haralick])

        # update the list of labels and feature vectors
        train_labels_final.append(label)
        train_global_features_final.append(global_feature)

rmf= RandomForestClassifier(n_estimators=100, random_state=9)
kfold = KFold(n_splits=10, random_state=9)
cv_results=cross_val_score(rmf,train_global_features_final,train_labels_final,cv=kfold,scoring="accuracy")
print(cv_results.mean(), cv_results.std())






# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100, random_state=9)
test_num=0
successes=0
# fit the training data to the model
clf.fit(train_global_features_final, train_labels_final)

for label in class_names: 
    for file in glob.glob('isolated/'+(label)+'/test/*.jpg'):
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, (512,512)) 
        
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        
        global_feature = np.hstack([fv_histogram, fv_haralick])
        #predict the label for the extracted features of each image
    
        predicted_label = clf.predict(global_feature.reshape(1,-1))
        #print(file.split("/")[1],predicted_label)
        if(file.split("/")[1]==predicted_label):
            successes+=1
        test_num+=1
print(successes/test_num)
