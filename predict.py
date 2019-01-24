import os
import sys

import numpy as np
import face_recognition

from PIL import Image
from keras.models import load_model


def parser():
    '''
    parse the command and call the functions

    -- first command
    - color
    - gray

    -- second command
    - path to the file you want to predict
    '''

    color_list = ['color', 'c', 'C', 'Color', 'colour', 'Colour']
    gray_list = ['gray', 'Gray', 'g', 'G', 'grey', 'Grey']

    try:
        if sys.argv[1] in gray_list:
            return predict_gray(sys.argv[2])
        else:
            return predict_color(sys.argv[2])
    except Exception as e:
        print(e)
        return 1

    return 0

def get_label_map():
    '''
    This assumes the
    '''
    label_map = {}
    folder_list = os.listdir('./test')

    for i, label in enumerate(folder_list, start=0):
        label_map[i] = label

    return label_map


# def predict_gray(path_to_image):
def predict_gray(path_to_image):
    '''
    Needs a label_map, and predict through that
    '''
    model = load_model('./model.h5')
    label_map = get_label_map()

    # load the image
    # image = face_recognition.load_image_file('./accuracytest/grayjasper2.jpg')
    image = face_recognition.load_image_file(path_to_image)
    encoding = face_recognition.face_encodings(image)

    if len(encoding) != 1:
        return "Failed"
    else:
        name = label_map[np.argmax(model.predict(np.array(encoding)))]

    return name


def import_predict_gray(model, path_to_image):
    label_map = get_label_map()

    # load the image
    # image = face_recognition.load_image_file('./accuracytest/grayjasper2.jpg')
    image = face_recognition.load_image_file(path_to_image)
    encoding = face_recognition.face_encodings(image)

    if len(encoding) != 1:
        # means that there is no proper faces recognised
        # return failed so that when saving the file, it is recongised that
        # at this time it failed
        return "Failed"
    else:
        name = label_map[np.argmax(model.predict(np.array(encoding)))]

    return name


def predict_color(path_to_image):
    '''
    Needs a label_map, and predict through that
    this function is to predict the pictures with colour
    Instead of using another model with the colour, we will just 
    turn the image to grayscle and predict through that
    '''

    model = load_model('./model.h5')
    label_map = get_label_map()

    # load the image with PIL and save it in grayscale
    img = Image.open(path_to_image).convert('LA')
    img.save('foo.png')

    # load the image in 
    image = face_recognition.load_image_file('foo.png')
    encoding = face_recognition.face_encodings(image)

    # model predict
    if len(encoding) != 1:
        if len(encoding) == 0:
            print('No face found')
        else:
            print('More than 1 face found')
        return "Failed"
    else:
        name = label_map[np.argmax(model.predict(np.array(encoding)))]
    
    # remove picture
    os.remove('foo.png')

    # return
    return name

def import_predict_color(model, path_to_image):
    label_map = get_label_map()

    # load the image with PIL and save it in grayscale
    img = Image.open(path_to_image).convert('LA')
    img.save('foo.png')

    # load the image in 
    image = face_recognition.load_image_file('foo.png')
    encoding = face_recognition.face_encodings(image)

    # model predict
    if len(encoding) != 1:
        if len(encoding) == 0:
            print('No face found')
        else:
            print('More than 1 face found')
        return "Failed"
    else:
        name = label_map[np.argmax(model.predict(np.array(encoding)))]
    
    # remove picture
    os.remove('foo.png')

    # return
    return name



if __name__ == "__main__":
    print(parser())
