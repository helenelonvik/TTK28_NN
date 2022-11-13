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
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten


def neuralNetwork(channels, samples, nb_classes):
    
    model = Sequential()
    model.add(Conv2D(32, (1,32), activation='relu', input_shape=(channels, samples, 1)))
    model.add(MaxPooling2D(1,8))
    model.add(Conv2D(16, (1,32), activation = 'relu'))
    model.add(MaxPooling2D(1,8))
    
    
    model.add(Flatten())
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(nb_classes))
    # Input     
    #input1   = Input(shape = (channels, samples, 1))


    # hidden layers
    #hidden1 = Conv2D(5, (1, 32), input_shape = (channels, samples, 1))(input1)
    #hidden1 = Maxpooling2D((2,2))
    #hidden2 = Dense(4, activation='sigmoid')(hidden1)

    # Output layer
    #output = Activation('softmax', name='softmax', input_shape=(32, 15, 2560, 5))(hidden1)

    #model = Model(inputs=input1, outputs=output)
    model.summary()
    return model 