import os

import numpy as np
import face_recognition

from PIL import Image
from keras.models import load_model


def predict_gray(path_to_image):
    '''
    Needs a label_map, and predict through that
    '''
    model = load_model('./model.h5')

    return 0


def predict_color(path_to_image):
    '''
    Needs a label_map, and predict through that
    this function is to predict the pictures with colour
    Instead of using another model with the colour, we will just 
    turn the image to grayscle and predict through that
    '''

    model = load_model('./model.h5')

    # load the image with PIL and save it in grayscale
    # load the image in 
    # model predict
    
    # remove picture

    # return

    return 0


if __name__ == "__main__":
    print(predict_color())
    # print(predict_gray))