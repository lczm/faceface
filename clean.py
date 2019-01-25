import os
import re
import utils
import face_recognition

from PIL import Image
from pprint import pprint

def check(dir):
    if os.path.isdir(dir):
        return True
    else:
        try:
            os.mkdir(dir)
            print('Made directory {}'.format(dir))
            return True
        except Exception as e:
            print('Error', e)
            return False

def replicate_rename(source_list, source, end):
    '''
    replicates the source_list and renames the source to end
    '''

    # clean source and end using regex
    source = re.sub(r'[^\w]', '', source)
    end = re.sub(r'[^\w]', '', end)

    print(source)
    print(end)

    replicated_list = []
    for dir_string in source_list:
        replicated_list.append(dir_string.replace(source, end))
    return replicated_list


def convertgray(source, end):
    '''
    Given a source directory, read through the source directory
    and for every image, return the requilvalent of the image in the end 
    directory in grayscale
    '''

    # checks for directory validity
    check(source)
    check(end)

    # do a os path walk on source
    # os walk -> (dirpath, dirnames, filenames)
    paths, dirnames, files = [], [], []

    for dirpath, dirname, filelist in os.walk(source):
        paths.append(dirpath)
        dirnames.append(dirname)
        files.append(filelist)

    # first one is always empty
    paths.pop(0)
    files.pop(0)
    dirnames = utils.flatten_list(dirnames)

    # clean files
    files = utils.clean_multiple_list(files)
    end_files = replicate_rename(paths, source, end)

    # make end directory if they do not exist
    for sub_dir in end_files:
        check(sub_dir)

    # len(paths) and len(files) are the same
    for i in range(len(paths)):
        for item in files[i]:
            '''
            load image for face_recognition
            if == 1 face locations, convert to grayscale and save
            '''

            load_image = face_recognition.load_image_file(os.path.join(paths[i], item))
            image_faces = face_recognition.face_locations(load_image)
            if len(image_faces) == 1:
                # open from source directory, paths
                # convert to grayscale
                img = Image.open(os.path.join(paths[i], item)).convert('LA')
                # print out where its saving
                print(os.path.join(end_files[i],  item))
                # save them all as pngs
                img.save(str(os.path.join(end_files[i], item)), "PNG")
                # img.save(str(os.path.join(end_files[i], item)) + 'png')

                # open up the image again
                second_image = face_recognition.load_image_file(os.path.join(end_files[i], item))
                second_faces = face_recognition.face_locations(second_image)
                if len(second_faces) != 1:
                    # os.system('rm {}.png'.format(os.path.join(end_files[i], item)))
                    print(os.path.join(end_files[i], item))
                    os.system('rm {}'.format(os.path.join(end_files[i], item)))
                
            else:
                print("Image does not have 1 face")
    
    return 0


if __name__ == "__main__":
    convertgray('./testceleb', './test')
