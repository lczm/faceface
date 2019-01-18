import os
import math
import random
import face_recognition

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import numpy as np

from functools import reduce
from pprint import pprint

from PIL import Image

def clean(unclean_list):
    clean_list = []
    for element in unclean_list:
        if '.jpg' in element or '.png' in element or '.jpeg' in element:
            clean_list.append(element)
        else:
            # if file doesnt exist, ignore
            pass

    return clean_list

def clean_dict(unclean_dict):
    clean_dict = {}
    for key, value in unclean_dict.items():
        if value == []:
            pass
        elif value == None:
            pass
        elif value == 0:
            pass
        else:
            clean_dict[key] = value
    return clean_dict

def load_data():
    '''
    Load the data from the file
    Directories
    - [Name]
        - image.jpg
        - image.jpg
        - image.jpg
        - image.jpg
    - [Name]
        - image.jpg
        - image.jpg
        - image.jpg
        - image.jpg
    '''
    map_dict = {}
    # test file and encoding

    # read file

    for root, _, files in os.walk(os.path.join('./testceleb')):
        map_dict[root] = clean(files)
    
    map_dict = clean_dict(map_dict)

    # directories
    keys = list(map_dict.keys())
    values = reduce(lambda x, y: x+y, list(map_dict.values()))

    print(keys[0])
    print(values[0])

    specific_value = map_dict[keys[0]]
    # specific_encoding = np.array([])
    specific_encoding = []

    # test image
    test_image = face_recognition.load_image_file(os.path.join(keys[1], values[271]))
    test_image_encoding = face_recognition.face_encodings(test_image)[0]

    test_image2 = face_recognition.load_image_file(os.path.join(keys[0], values[2]))
    test_image_encoding2 = face_recognition.face_encodings(test_image2)[0]

    counter = 0

    for image in specific_value:
        load_image = face_recognition.load_image_file(os.path.join(keys[0], image))
        location = face_recognition.face_locations(load_image)
        if len(location) == 1:
            print(counter)
            # encode face
            encoding = face_recognition.face_encodings(load_image)[0]
            # np.append(specific_encoding, encoding)
            specific_encoding.append(encoding)
            counter+=1
    
    pprint(specific_encoding)

    flatten_encoding = np.array(specific_encoding).flatten()

    print(np.std(flatten_encoding))
    print(np.var(flatten_encoding))
    print(np.mean(flatten_encoding))


    face_distances = face_recognition.face_distance(specific_encoding, test_image_encoding)
    for i, face_distance in enumerate(face_distances):
        if face_distance < 0.6:
            print('abrahim', i, 'This is abraham')
        else:
            print('abrahim', i, 'No')

    print('----------------------')

    total = 0
    correct = 0
    
    correct_face_distance = face_recognition.face_distance(specific_encoding, test_image_encoding2)
    for i, face_distance in enumerate(correct_face_distance):
        if face_distance < 0.6:
            print('abrahim', i, 'This is abraham')
            correct+=1
        else:
            print('abrahim', i, 'No')
        total+=1

    print('total', total)
    print('correct', correct)

    print('percentage right', correct / total)


    return 0


def model():
    '''
    Make a keras model and returns it
    '''
    return 0


def main():
    return 0


if __name__ == "__main__":
    load_data()
    # main()