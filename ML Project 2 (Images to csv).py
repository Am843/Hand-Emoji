import cv2
import numpy as np
import os
import pandas as pd

root = './Gestures'

for directory, subdirectories, files in os.walk(root):
    for file in files:
        print(file)
        im = cv2.imread(os.path.join(directory, file))
        value = im.flatten()
        value = np.hstack((directory[8:], value))
        df = pd.DataFrame(value).T
        df = df.sample(frac=1)
        with open('training data.csv', 'a') as dataset:
            df.to_csv(dataset, header=False, index=False)