import keras
import os

import numpy as np
import tensorflow as tf
import face_recognition
from multiprocessing import Pool, cpu_count

from pprint import pprint
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# from keras.layers import Conv1D, Activation, MaxPool1D, Flatten, Dense, Dropout
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical

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

# Encodes the image at the given path
# Returns the encoding for the image
def encode_image(path):
    print("encoding : ", path, flush=True)
    # open the iamge and encode it, afterwards
    # attaching it to the image_to_label_map
    load_image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(load_image)
    
    # print(len(encoding), flush=True)
    if len(encoding) != 1:
        return -1

    return encoding


def load_data(directory):
    '''
    directory here refers to the directory where the images can be found for training
    '''

    # getting the folder_list and the labels as length of folder list
    folder_list = os.listdir(directory)
    total_labels = len(folder_list)

    '''
    label_map = {
        1 : 'label',
        2 : 'label',

        'label' : 1,
        'label' : 2,
        ...
    }
    '''
    label_map = {}
    # for i, label in enumerate(folder_list, start=1):
    #     label_map[i] = label

    for i, label in enumerate(folder_list, start=0):
        label_map[label] = i
        # label_map[i] = label

    encoding_sequence = []
    label_sequence = []

    # compile dataset image paths
    image_paths = []
    for root, dirs, files in os.walk(os.path.join(directory)):
        for file in files:
            image_paths.append(os.path.join(root, file))

            # Extract label for image at path
            # label = label_map[extract_substring(root)]
            # label_sequence.append(label)

            label_sequence.append(extract_substring(root))
    
    proc_pool = Pool(cpu_count())
    encoding_sequence = proc_pool.map(encode_image, image_paths)

    clean_encoding = []
    clean_label = []

    # print('test')
    # for i in range(len(encoding_sequence)):
    #     print(encoding_sequence[i])

    # clean the list of encodings, labels again
    # at this stage, the len(encoding_sequence) and len(label_sequence) are the same
    if len(encoding_sequence) == len(label_sequence):
        for i in range(len(encoding_sequence)):
            # if len(encoding_sequence[i]) == 1:
            if encoding_sequence[i] == -1:
                pass
            else:
                # [0] because face_encodings returns a list and we only take the first one
                clean_encoding.append(encoding_sequence[i][0])
                clean_label.append(label_sequence[i])
    else:
        print('something went wrong')
        return 1

    # convert the encoding and label sequences to a numpy array before splittin
    # them up and passing them into the fit function


    int_clean_label = []
    for label in clean_label:
        int_clean_label.append(label_map[label])

    clean_encoding = np.array(clean_encoding)

    # one hot encode the labels
    # clean_label = to_categorical(np.array(clean_label))
    int_clean_label = to_categorical(np.array(int_clean_label))

    print('length of keys', len(label_map.keys()))
    # print(len(set(list(int_clean_label.tolist()))))
    # print(set(int_clean_label.tolist()))

    # for i in range(len(clean_label)):
    #     print('label', clean_label[i].shape)

    # x_train, x_test, y_train, y_test = train_test_split(clean_encoding, clean_label, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(clean_encoding, int_clean_label, test_size=0.2)

    # print('x_train', len(x_train.shape))
    # print('x_test', len(x_test.shape))
    # print('y_train', len(y_train.shape))
    # print('y_test', len(y_test.shape))

    # print out the shapes of the arrays if needed
    return (x_train, y_train), (x_test, y_test), total_labels

    # return 0


def generate_dense_mode():
    (x_train, y_train), (x_test, y_test), num_labels= load_data('./test')

    print('number of labels', num_labels)

    # print(x_train.shape)
    # print(len(y_train))
    # print(y_train.shape)

    # count the number of labels in y_train
    # print('length', len(y_train.unique))

    pprint(y_train)

    '''
    Just add a bunch of normal dense layers without convolution to work
    '''

    model = Sequential()

    # starting layer
    model.add(Dense(128,activation='relu', input_shape=(128,)))

    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # ending layer
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    # model.fit(x_train, y_train, batch_size=2, epochs=30)
    # no batch size
    model.fit(x_train, y_train, batch_size=15,epochs=100, validation_data=(x_test, y_test))

    # (eval_loss, eval_accuracy) = model.evaluate(
    #     y_train , y_test, batch_size=15, verbose=1)

    # print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    # print("[INFO] Loss: {}".format(eval_loss))

    model.save('model.h5')

    # use np.argmax to inverse the prediction

    return model


if __name__ == "__main__":
    print(generate_dense_mode())
    # print(load_data('./test'))
