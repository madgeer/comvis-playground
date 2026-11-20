import cv2
import numpy as np
from os import listdir
from os.path import isfile, isdir, join

def load_with_pad(lokasi, mode=1, size=(224, 224)):
    """Load gambar dari folder, resize dengan menjaga aspect ratio, lalu padding (letterbox)."""
    dirlist = listdir(lokasi)
    data = []
    label = []
    target_w, target_h = size

    for classdir in dirlist:
        class_path = join(lokasi, classdir)
        if not isdir(class_path):
            continue
        for filename in listdir(class_path):
            file_loc = join(class_path, filename)
            print(file_loc)
            img = cv2.imread(file_loc, mode)
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if mode == 0:
                canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            else:
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img

            data.append(canvas)
            label.append(classdir)

    return (data, label)


def load(lokasi, mode=1, size=None):
    dirlist = listdir(lokasi)
    data = []
    label = []
    for (i, classdir) in enumerate(dirlist):
        listfile = listdir(join(lokasi, classdir))
        for (j, filename) in enumerate(listfile):
            file_loc = join(lokasi, classdir, filename)
            print(file_loc)
            img = cv2.imread(file_loc, mode)
            if size != None:
                img = cv2.resize(img, size)

            data.append(img)
            label.append(classdir)

    return (data, label)
