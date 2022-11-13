import numpy as np
from mne.filter import notch_filter, filter_data

def extractAndFilter(all_data):
    data = all_data['X']
    classes = all_data["y"][0]
    markers = all_data["markers"][0]   
    data = np.transpose(data) 
    #print(markers)
    #print(data.shape)
    raw_task_data = []
    tags = []

    classes = np.array(classes)-1  # must do it because NN wants classes 0 and 1, but we got 1 and 2

    electrodes = [i for i in range(0,15)]

    freq = 512
    task_time = 5
    task_sample_length = freq*task_time

    for i, mark in enumerate(markers):

        task_data = data[electrodes, int(mark+(2.5*freq)):int(mark+(2.5*freq))+task_sample_length]
        # brain imegary starter etter 2.5 sekunder

        #filtered_data = notch_filter(data, freq, 50, filter_length=50, method='spectrum_fit')
        filtered_data = filter_data(task_data, sfreq=freq, l_freq=4, h_freq=30, filter_length=10, method='iir')
        #Explain why we remove under 4 and over 30. Under 4 is "doing nothing"

        raw_task_data.append(filtered_data)
        tags.append(classes[i])

        
    return raw_task_data, tags