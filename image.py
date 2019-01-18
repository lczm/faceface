import os
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def folders():
    # list down all the current folder directory
    folder_dir = './lfw/'
    folders = os.listdir(folder_dir)
    # go into each folder
    for folder in folders[0:3]:
        sub_dir = folder_dir + str(folder)
        # get all the images from the folder
        subfolder = os.listdir(sub_dir)
        for item in subfolder:
            # display the images in matplotlib
            img = mpimg.imread(sub_dir + '/' + item)
            plt.figure()
            plt.imshow(img)
            plt.show()



if __name__ == "__main__":
    folders()
    pass
