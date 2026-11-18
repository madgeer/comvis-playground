import cv2
from os import listdir
from os.path import isfile, join
def load(lokasi,mode=1,size=None):
    dirlist = listdir(lokasi)
    data = []
    label = []
    for(i,classdir) in enumerate(dirlist):
        listfile = listdir(join(lokasi,classdir))

        for(j,filename) in enumerate(listfile):
            file_loc = join(lokasi,classdir,filename)
            print(file_loc)
            img = cv2.imread(file_loc,mode)
            if size!=None:
                img = cv2.resize(img,size)

            data.append(img)
            label.append(classdir)

    return (data, label)