import pandas as pd
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
import shutil



def create_directory(path, directory):
    def segmentation(path):
        csv = pd.read_csv(path)
        csv_data = csv.iloc[:,0]
        data = np.array(csv_data)
        signals = []
        count = 1
        peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 250)[0]
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1]- i)
            x = peaks[count - 1] + diff1//2
            y = peaks[count + 1] - diff2//2
            signal = data[x:y]
            signals.append(signal)
            count += 1
        return signals
    
    def signal_to_img(array, directory):  
        while(1):
            if os.path.exists(directory):
                shutil.rmtree(directory)
            else:
                os.makedirs(directory)
                break

        for count, i in enumerate(array):
            fig = plt.figure(frameon=False)
            plt.plot(i) 
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            filename = directory + '/' + str(count)+'.png'
            fig.savefig(filename)
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)
        return directory
    array = segmentation(path)
    directory = signal_to_img(array, directory)
    return directory
    

    

