import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from neuralNetwork import neuralNetwork
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image
import pandas as pd
import cv2

# example of loading the mnist dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam

from matplotlib import pyplot as plt
"""
prøv med 32 batch og
Change outputlayer

"""

def getData(type='ST'):
    if type=='MNIST':
        print('MNIST')
        img_size=28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((-1, img_size, img_size, 1))
        x_test = x_test.reshape((-1, img_size, img_size, 1))
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

        
        # one hot encode target values
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        #y_test = to_categorical(y_test)

        
        # convert from integers to floats
        train_norm = x_train.astype('float32')
        test_norm = x_test.astype('float32')
        val_norm = x_val.astype('float32')
        # normalize to range 0-1
        x_train = train_norm / 255.0
        x_test = test_norm / 255.0
        x_val = val_norm /255.0

        model = neuralNetwork(img_size, 10)
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        
        model.compile(loss='categorical_crossentropy',          #binary_crossentropy' hvis 2 klasser   sparse_categorical_crossentropy, hvis flere klasser men ikke en hot
                    optimizer=SGD(learning_rate=0.01, momentum=0.9), 
                    metrics=['accuracy'])
        fmodel = model.fit(x_train, y_train, epochs=15, callbacks=[callback], validation_data=(x_val,y_val))
        predictions = model.predict(x_test)
        #print(predictions)
        preds = predictions.argmax(axis=-1)
        #print(preds)
        #print(y_test)


        """y_pred =[]
        for i in range(len(predictions)):
            y = animals[preds[i]]
            y_pred.append(y)

        print(fmodel.history.keys())
        """
        print("classification accuracy: ", str(round(accuracy_score(y_test,preds)*100,3)), "%")
        print("f1 score ", str(round(f1_score(y_test,preds, average='micro')*100,3)), "%")
        plt.plot(fmodel.history['accuracy'])
        plt.plot(fmodel.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(fmodel.history['loss'])
        plt.plot(fmodel.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        """
        pd.DataFrame(fmodel.history).plot(figsize=(8,5))
        plt.show()

        #"""


    else:
        print('Animals')

        animals = ['cane','elefante','pecora']#,'farfalla']
        animal_classes = [0,1,2,3,4]
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
                
        
            tags = [animal for _ in range(len(files))]#[animal_classes[j] for i in range(len(files))]
            #pictures+=files
            classes+=tags

    
        #print(pictures.shape)


        print(len(classes))
        print(len(pictures))
        
        pictures, classes = shuffle(pictures,classes, random_state=0)

        print('shuffeled')
        x_train, x_test, y_train, y_test = train_test_split(pictures, classes, test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) #0.25*0.8 = 0.2

    
        #normalize data
        x_train = np.array(x_train).astype('float32')/255.0
        x_test = np.array(x_test).astype('float32')/255.0
        x_val= np.array(x_val).astype('float32')/255.0

        print('normalized')

        x_train.reshape(-1,img_size, img_size,1)
        x_test.reshape(-1, img_size,img_size,1)
        x_val.reshape(-1,img_size,img_size,1)
        
        #y_test2 = y_test
        print('reshape')
        #df = pd.DataFrame({
        #    'Animal_test': y_test
        #})
        da = pd.DataFrame({ 
            'Animal_train': y_train
        })
        dt = pd.DataFrame({
            'a':y_val
        })
        #y_test = pd.get_dummies(df['Animal_test'])
        y_train = pd.get_dummies(da['Animal_train'])
        y_val = pd.get_dummies(dt['a'])
        

        # Can add data augmentation for the test data
        model = neuralNetwork(img_size, len(animals))
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        
        model.compile(loss='categorical_crossentropy',          #binary_crossentropy' hvis 2 klasser   sparse_categorical_crossentropy, hvis flere klasser men ikke en hot
                    optimizer=Adam(learning_rate=0.001),#SGD(learning_rate=0.01, momentum=0.9), 
                    metrics=['accuracy'])
        fmodel = model.fit(x_train, y_train, epochs=30, validation_data=(x_val,y_val), batch_size=32)  #høyere batch size lavere loss men dårliger accuracy  lavere enn 32 gir lik acc, men dårligere og mere oscillerende loss
        predictions = model.predict(x_test)
        #print(predictions)
        preds = predictions.argmax(axis=-1)
        #print(preds)
        #print(y_test)


        y_pred =[]
        for i in range(len(predictions)):
            y = animals[preds[i]]
            y_pred.append(y)

        print(fmodel.history.keys())

        print("classification accuracy: ", str(round(accuracy_score(y_test,y_pred)*100,3)), "%")
        print("f1 score ", str(round(f1_score(y_test,y_pred, average='micro')*100,3)), "%")
        plt.plot(fmodel.history['accuracy'])
        plt.plot(fmodel.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(fmodel.history['loss'])
        plt.plot(fmodel.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        """pd.DataFrame(fmodel.history).plot(figsize=(8,5))
        plt.show()

        #"""