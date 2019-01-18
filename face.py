import os
import joblib
import random
import face_recognition

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

import tensorflow as tf
import numpy as np

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from PIL import Image
from pprint import pprint


def format_tuple(tuple_value):
    new_tuple = (tuple_value[3], tuple_value[0], tuple_value[1], tuple_value[2])
    return new_tuple

def path(dir):
    # './lfw/'
    # key  - value
    # root - files
    map_dict = {}
    for root, a, files in os.path.join(dir):
        print('root', root)
        print('a', a)
        print('files', files)
        map_dict[root] = files

    # pprint(map_dict)
    return map_dict

def check_size():
    file_dict = path(os.path.join('./lfw/'))
    widths = []
    heights = []
    for key, values in file_dict.items():
        for value in values:
            image = Image.open(os.path.join(str(key) + '/' + str(value)))
            width, height = image.size
            widths.append(width)
            heights.append(height)

    widths = list(set(widths))
    heights = list(set(heights))

    if len(widths) == 1 and len(heights) == 1:
        print('All images have the same size')
        return 0
    else:
        print('Variation in image sizes')
        print('Width  : ', widths)
        print('Height : ', heights)
        return 1


def testgridspec():
    '''
    Function to show a 3x3 image in pyplot
    To show whether the images are being processed or not
    '''

    x = 0
    y = 0
    flag = True

    fig = plt.figure(constrained_layout=True)
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # load image
    image = Image.open('./flum2.jpg')

    while True:

        print(x, y)

        if x == 2 and y == 2:
            if flag == True:
                plot = fig.add_subplot(grid[x, y])
                plot.axes.get_yaxis().set_visible(False)
                plot.imshow(image)
                flag = False
            else:
                print('Not plotting')
                # break instead of pass, as not in a bigger loop
                break
        elif x == 0 and y == 0:
                plot = fig.add_subplot(grid[x, y])
                plot.axes.get_yaxis().set_visible(False)
                plot.imshow(image)
                y += 1
        elif y == 2:
            plot = fig.add_subplot(grid[x, y])
            plot.axes.get_yaxis().set_visible(False)
            plot.imshow(image)
            y = 0
            x += 1
        else:
            plot = fig.add_subplot(grid[x, y])
            plot.axes.get_yaxis().set_visible(False)
            plot.imshow(image)
            y += 1


    plt.show()
    return 0


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


def keras_model():
    '''
    model to learn and predict
    '''

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()

    # adding of layers
    model.add(tf.keras.layers.Flatten(input_shape=x_train[0].shape))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # number in this layer refers to the number of output
    # in this case, 2 as it is either 'authorized' or 'not authorized'
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    # use sigmoid for this as it is a yes/no result
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.fit(x_train, y_train,batch_size=32, epochs=1)

    model.summary()
    print(model)


def knn_model():
    '''
    find the files and classify them according to their name
    '''

    person_dict = path(os.path.join('./testceleb'))
    person_dict = clean_dict(person_dict)
    # pprint(person_dict)

    # person's image
    humans = []
    # names
    labels = []
    
    pprint(person_dict)

    for key, values in person_dict.items():
        for value in values:
            # humans.append(Image.open(os.path.join(str(key) + '/' + str(value))))
            # load image
            image_encoding = face_recognition.face_encodings(os.path.join(str(key) + '/' + str(value)))
            # append it
            humans.append(image_encoding)
            labels.append(key)

    print(len(person_dict))
    # pprint(person_dict)

    x_train, x_test, y_train, y_test = train_test_split(humans, labels, test_size=0.2)
    # pprint(x_train)
    # pprint(y_train)

    model = neighbors.KNeighborsClassifier(n_neighbors=5)
    return model


def main():
    file_dict = path('./lfw/')

    # setting up the gridspec for plotting
    fig = plt.figure(constrained_layout=True)
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    temp_faces = []
    i = 0
    x = 0
    y = 0
    flag = True

    for key, values in file_dict.items():
        print('key', key)
        print('value', values)
        if i > 8:
            break
        '''
        key -> directory of file
        value -> file itself
        '''
        # print(key, value)
        for value in values:
            image = face_recognition.load_image_file(os.path.join(str(key) + '/' + str(value)))
            face_locations = face_recognition.face_locations(image)
            print(face_locations)
            # if there are faces, load the image
            if len(face_locations) > 0:
                # pilimage = Image.open(os.path.join(str(key) + '/' + str(value)))
                print(str(key) + '/' + str(value))
                pilimage = Image.open(str(key) + '/' + str(value))
                for faces in face_locations:
                    cropped = pilimage.crop(format_tuple(faces))

                    if x == 2 and y == 2:
                        if flag == True:
                            plot = fig.add_subplot(grid[x, y])
                            plot.axes.get_yaxis().set_visible(False)
                            plot.imshow(image)
                            flag = False
                        else:
                            print('Not plotting')
                            pass
                    elif x == 0 and y == 0:
                            plot = fig.add_subplot(grid[x, y])
                            plot.axes.get_yaxis().set_visible(False)
                            plot.imshow(image)
                            y += 1
                    elif y == 2:
                        plot = fig.add_subplot(grid[x, y])
                        plot.axes.get_yaxis().set_visible(False)
                        plot.imshow(image)
                        y = 0
                        x += 1
                    else:
                        plot = fig.add_subplot(grid[x, y])
                        plot.axes.get_yaxis().set_visible(False)
                        plot.imshow(image)
                        y += 1

        print('counter', i)
        i+=1

    plt.show()

if __name__ == "__main__":
    # main()
    # testgridspec()
    # check_size()
    # load_data()

    knn_model()

