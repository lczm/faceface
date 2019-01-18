import keras
import os

import numpy as np
import tensorflow as tf
import face_recognition

from pprint import pprint
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import svm

# from keras.layers import Conv1D, Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.layers import Conv1D, Activation, MaxPool1D, Flatten, Dense, Dropout
from keras.models import Sequential

'''
This file will attach each of the image from the folder that the
'clean.py' file has completed

Attaching that to its individual label and returning a model to an interface
that can be used
'''

def extract_substring(dir_str):
    '''
    This function will return the substring
    after the last \ in a directory string.
    '''
    # print('dir_str', dir_str)
    index_list = []
    for letter in dir_str:
        # windows
        # if letter == str('\\'):
        # linux
        # if letter == str('/'):
        #     index_list.append(dir_str.index(letter))

        # updated version
        for i in range(len(dir_str)):
            if dir_str[i] == '/':
                    index_list.append(i)

    filtered = dir_str[index_list[len(index_list)-1]:]
    # pop out the first letter
    filtered = filtered[1:]
    return filtered


def load_data(directory):
    '''
    directory here refers to the directory where the images can be found for training
    '''

    # getting the folder_list and the labels as length of folder list
    folder_list = os.listdir(directory)
    total_labels = len(folder_list)
    # print(total_labels)

    # can use one hot encoding for the label
    '''
    label_map = {
        1 : 'label',
        2 : 'label',
        ...
    }
    '''
    label_map = {}
    # for i, label in enumerate(folder_list, start=1):
    #     label_map[i] = label

    for i, label in enumerate(folder_list, start=1):
        label_map[label] = i

    encoding_sequence = []
    label_sequence = []

    for root, dirs, files in os.walk(os.path.join(directory)):
        for file in files:
            print(root, file)
            # open the iamge and encode it, afterwards
            # attaching it to the image_to_label_map
            load_image = face_recognition.load_image_file(os.path.join(root, file))
            encoding = face_recognition.face_encodings(load_image)

            # TODO : check for the label validity with the folder_list, add exception
            encoding_sequence.append(encoding)
            label_sequence.append(label_map[extract_substring(root)])


    # pprint(label_map)

    # pprint(encoding_sequence)
    # pprint(label_sequence)
    print('encoding sequence', len(encoding_sequence))
    print('label sequence', len(label_sequence))

    # for checking purposes
    # if len(label_sequence) == len(encoding_sequence):
    #     for i in range(len(label_sequence)):
    #         print(encoding_sequence[i], label_sequence[i])

    # conver the encoding and label sequences to a numpy array before splittin
    # them up and passing them into the fit function
    encoding_sequence = np.array(encoding_sequence)
    label_sequence = np.array(label_sequence)

    # testing purposes
    print(encoding_sequence[0])
    print(len(encoding_sequence))
    print(type(encoding_sequence))

    x_train, x_test, y_train, y_test = train_test_split(encoding_sequence, label_sequence, test_size=0.2)
    # print out the shapes of the arrays if needed
    return (x_train, y_train), (x_test, y_test), total_labels
    # return 0


def generate_svm_model():
    '''
    build up a svm model from scikitlearn and use that to predict
    faces
    '''

    (x_train, y_train), (x_test, y_test), num_labels= load_data('./test')

    model = svm.SVC(kernel='linear')
    print('model at the first stage', model)
    # fitting the model
    model.fit(x_train, y_train)
    print(model)
    return model


def generate_dense_mode():
    (x_train, y_train), (x_test, y_test), num_labels= load_data('./test')

    '''
    Just add a bunch of normal dense layers without convolution to work
    '''

    model = Sequential()
    model.add(Dense())
    model.add(Dense())
    model.add(Dense())
    model.add(Dense())
    model.add(Dense())
    model.add(Dense())

    # ending layer
    model.add(Dense(num_labels, activation='softmax'))
    return model


def generate_model():

    # standard parametes for machine learning
    # img_shape = (28, 28, 1)
    # num_labels = get_labels()
    # num_labels = 10

    (x_train, y_train), (x_test, y_test), num_labels= load_data('./test')

    encode_amount = np.shape(x_train)

    '''
    test 32 -> 64 -> 128 -> Flatten -> 32 -> 64 -> 128 -> softmax
    '''

    model = Sequential()

    # kernel_size = 2d (x, y) area of the convolutional lens

    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(encode_amount, 1)))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) 
    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) 

    # model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=(encode_amount, 1)))
    model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=(None)))
    model.add(Conv1D(64, kernel_size=5, activation='relu'))
    model.add(Conv1D(128, kernel_size=5, activation='relu'))

    # helps prevent overfitting
    # model.add(Dropout(0.5)) 

    # model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())

    # model.add(Dense(128, activation='relu')) 

    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    # model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

    model.fit(x_train, y_train,batch_size=32, epochs=3)

    model.summary()

    return model


if __name__ == "__main__":
    print(generate_model())
    # generate_svm_model()
    # print(load_data('./test'))

    # print(extract_substring(r'./test\nicholas cage'))
