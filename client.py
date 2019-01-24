import os
import cv2

import predict

from datetime import datetime
from keras.models import load_model

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def check_csv():
    '''
    makes sure the csv exists
    '''
    if not os.path.isfile('./log.csv'):
        touch('./log.csv')
    
    return 0

def load_my_model():
    model = load_model('model.h5')
    return model


def cv_window():
    '''
    initialises the webcam and 
    allows the user to press space to 
    [predict]
    - whats actually happening is that pressing space saves the picture
    '''

    model = load_my_model()

    cv2.namedWindow("faceface")
    vc = cv2.VideoCapture(0)

    # initialize i = 0
    i = 0
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    with open('./log.csv', 'w') as csvfile:
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
            if key == ord(' '):
                # save the frame as grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                date_string = './livetest/' + f'{datetime.now():%Y-%m-%d-%H:%M:%S%z}' + '.jpg'
                # write the file to disc
                cv2.imwrite(date_string, frame)
                # predict
                predict_value = predict.import_predict_gray(model, date_string)

                print('predict_value', predict_value)

                # Here, the location can also be decided.
                # However, as just a concept, the location can be hardcoded in
                # csvfile.write("{}, {}, {} \n".format(date_string, predict_value))
                csvfile.write("{}, {}, #BLK31-05-01 \n".format(date_string, predict_value))


    cv2.destroyWindow("faceface")
    vc.release()


if __name__ == "__main__":
    check_csv()
    cv_window()
