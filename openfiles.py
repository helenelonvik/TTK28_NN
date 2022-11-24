import glob
import os
import numpy as np
from mne.utils import set_log_level
import scipy.io as sio
from extractAndFilter import extractAndFilter
from sklearn.model_selection import train_test_split
from neuralNetwork import neuralNetwork

set_log_level(verbose=False)  # Prevent mne.CSP from printing a lot of stuff

def openfiles():
    print("opening files")

    data = []
    tags = []

    file_list = glob.glob(os.path.join("C:", os.sep, "Users", "helen", "Documents","TTK28_NN","S03","sessions",'*.mat'))
    #C:\Users\helen\Documents\TTK28_NN\S03

    for file in file_list:
        #all_data = np.load(file, allow_pickle=True)
        all_data = sio.loadmat(file)
        datas, classes = extractAndFilter(all_data)
        
        data += datas
        tags += classes

        print(np.array(data).shape, len(tags))

    # Call NN
    print('CNN')
    x_train, x_test, y_train, y_test = train_test_split(data, tags, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) #0.25*0.8 = 0.2

    #siden vi har classene 1 0, er disse allerede "one-hot" formatert. Hadde vi hatt 3 klasser hadde det vært nødvendig

    x_train = np.array(x_train).reshape(len(x_train),15,2560,1) # markers, channels, samples, kernel
    #print(x_train.shape)
    x_val = np.array(x_val).reshape(len(x_val),15,2560,1)
    x_test = np.array(x_test).reshape(len(x_test),15,2560,1)

    model = neuralNetwork(15, 2560, 2)

    model.compile(loss='mse', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    fmodel = model.fit(x_train, np.array(y_train), epochs=100, validation_data=(x_val,np.array(y_val)))
    predictions = model.predict(x_test)
    preds = predictions.argmax(axis=-1)
    correct = 0
    for i in range(len(preds)):
        if preds[i]==y_test[i]:
            correct +=1
    acc         = correct/len(y_test)*100
    print("Classification accuracy: ", str(acc))
    #print(predictions)