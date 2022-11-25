#  If we have small data, 
# running a large number of iteration can result in overfitting.

# https://towardsdatascience.com/breaking-the-curse-of-small-data-sets-in-machine-learning-part-2-894aa45277f4


# you may increase the number of layers, neurons or epochs. Theoretically those things may help improving model accuracy.
# https://becominghuman.ai/sequential-vs-functional-model-in-keras-20684f766057


import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D,BatchNormalization

#Add droppout
def neuralNetwork(img_size, nb_classes, dropout_rate =0.5):
    
    model = Sequential()

    #layer 1
    model.add(Conv2D(32,(11,11), activation='relu',padding="same", input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    #Pooling leads to dimensionality reduction, and implicit translation and rotational invariance
    # Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension.
    
    model.add(Conv2D(64, (5,5), padding="same", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()
   
    return model 

