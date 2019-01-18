import os


def label(source):
    '''
    Goes to the specific (source) directory
    and performs an os_walk while using the directory itself
    as the label

    directory structure

    source
        -sub-dir(person) [label]
            -pic1.jpg
        -sub-dir(person) [label]
            -pic1.jpg
        -sub-dir(person) [label]
            -pic1.jpg
    '''
    labels = []
    for root, dirs, files in os.walk(source):
        print(root, dirs, files)

    return 0



if __name__ == "__main__":
    label('./test')
