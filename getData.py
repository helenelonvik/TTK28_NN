import glob
import os
import numpy as np
from mne.utils import set_log_level
import scipy.io as sio
from extractAndFilter import extractAndFilter
from sklearn.model_selection import train_test_split
from neuralNetwork import neuralNetwork
from tensorflow.keras import utils as np_utils
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image
import pandas as pd
import cv2

"""
prøv først med img_size = 100
så batch size = 64
"""
"""
Resultater
"""
def getData():
    print('Animals')

    animals = ['cane','elefante']#,'farfalla','pecora']
    animal_classes = [1,2,3,4]
    classes = []
    pictures = []

    img_size = 100 #300
    for j, animal in enumerate(animals):
        files = glob.glob(os.path.join("C:", os.sep, "Users", "helenetl", "Documents","NN","TTK28_NN","raw-img",animal,'*.jpeg'))
        #print(files)
        for file in files:
            img_arr = cv2.imread(file)[...,::-1]
            #Reshape
            # print(img_arr.shape) # høyde, bredde, RGB
            resized = cv2.resize(img_arr, (img_size,img_size))
            pictures.append(resized)
            
       
        tags = [animal_classes[j] for i in range(len(files))]
        #pictures+=files
        classes+=tags

   
    #print(pictures.shape)


    print(len(classes))
    print(len(pictures))
    
    pictures, classes = shuffle(pictures,classes, random_state=0)

    print('shuffeled')
    x_train, x_test, y_train, y_test = train_test_split(pictures, classes, test_size=0.2, random_state=42)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) #0.25*0.8 = 0.2

    #normalize data
    x_train = np.array(x_train)/255
    x_test = np.array(x_test)/255
    x_val= np.array(x_val)/255

    print('normalized')
    x_train.reshape(-1,img_size, img_size,1)
    x_test.reshape(-1, img_size,img_size,1)
    x_val.reshape(-1,img_size,img_size,1)

    print('reshape')
    #y_train      = np_utils.to_categorical(np.array(y_train))
    #y_val        = np_utils.to_categorical(np.array(y_val))
    y_test2 = y_test
    #y_test       = np_utils.to_categorical(np.array(y_test))
    y_test = np.array(y_test)-1
    y_train = np.array(y_train)-1
    y_val = np.array(y_val)-1

    # Can add data augmentation for the test data
    model = neuralNetwork(img_size, len(animals))
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    fmodel = model.fit(x_train, y_train, epochs=40 ,batch_size=64, callbacks=[callback])
    predictions = model.predict(x_test)
    preds = predictions.argmax(axis=-1)
    print(preds)

    correct = 0
    for i in range(len(preds)):
        if preds[i]==y_test2[i]-1:
            correct +=1
    acc         = correct/len(y_test)*100
    print("Classification accuracy: ", str(acc))

    print(np.array(x_train).shape)

