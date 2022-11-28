

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

def neuralNetwork(img_size, nb_classes):
    
    model = Sequential()

    #layer 1
    model.add(Conv2D(32,3, activation='relu', kernel_initializer='he_uniform',padding="same", input_shape=(img_size, img_size, 3)))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    
    #layer 2
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))


    #outputlayer
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu', kernel_initializer='he_uniform')) # Change to 30 / 300?
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()
   
    return model 

