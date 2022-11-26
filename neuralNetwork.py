#  If we have small data, 
# running a large number of iteration can result in overfitting.

# https://towardsdatascience.com/breaking-the-curse-of-small-data-sets-in-machine-learning-part-2-894aa45277f4


# you may increase the number of layers, neurons or epochs. Theoretically those things may help improving model accuracy.
# https://becominghuman.ai/sequential-vs-functional-model-in-keras-20684f766057


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

#Add droppout
def neuralNetwork(img_size, nb_classes):
    
    model = Sequential()

    #layer 1
    model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform',padding="same", input_shape=(img_size, img_size, 1)))
    model.add(MaxPooling2D(2,2))
    #model.add(Dropout(0.5))
    
    #layer 2
    #model.add(Conv2D(64, 3, padding="same", activation="relu"))
    #model.add(MaxPooling2D())
    #model.add(Dropout(0.5))


    #outputlayer
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu', kernel_initializer='he_uniform')) # Change to 30 / 300?
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()
   
    return model 

