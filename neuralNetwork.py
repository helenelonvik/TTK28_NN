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
def neuralNetwork(channels, samples, nb_classes, dropout_rate =0.5):
    
    model = Sequential()

    #layer 1
    model.add(Conv2D(8, (11,11), activation='relu',padding='same', input_shape=(channels, samples, 1)))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(1,8))
    model.add(AveragePooling2D((1,4) ))
    model.add(Dropout(dropout_rate))

    #layer 2
    model.add(Conv2D(16, (1,channels), activation = 'relu' ,padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(1,8))
    model.add(AveragePooling2D((1,4) ))
    model.add(Dropout(dropout_rate))
    
    """#layer 2
    model.add(Conv2D(64, (1,32), activation = 'relu' ,padding='same'))
    model.add(MaxPooling2D(1,8))
    model.add(Dropout(dropout_rate))
    """
    
    model.add(Flatten())
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(nb_classes))

    model.summary()
   
    return model 