# -*- coding: utf-8 -*-
# The script is to detect face and then align the photo.

import os
import traceback

from PIL import Image

import cv2
import numpy as np


class IDCropper(object):
    def __init__(self, x1=-2, x2=1.5, y1=1.4, y2=2.15):
        cur_dir = os.path.dirname(__file__)
        path = os.path.join(cur_dir, 'model/haarcascade_frontalface_default.xml')
        self.faceCascade = cv2.CascadeClassifier(path)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def crop(self, gray):
        """
        Crop out a bounding box of ID number from grayscale img
        :param gray:
        :return: cropped img containing ID number
        """
        assert len(gray.shape) == 2

        img_h, img_l = gray.shape

        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        x, y, w, h = 0, 0, -1, -1

        for (x1, y1, w1, h1) in faces:
            if w1 > w and h1 > h:
                x, y, w, h = x1, y1, w1, h1

        roi_x1 = max(0, x + int(self.x1 * w))
        roi_x2 = min(img_l-1, x + int(self.x2 * w))
        roi_y1 = max(0, y + int(self.y1 * h))
        roi_y2 = min(img_h-1, y + int(self.y2 * h))

        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            # Error: Detected face is not in reasonable positions.
            return None

        roi = cv2.resize(roi, (560, 120), interpolation=cv2.INTER_CUBIC)
        return roi

if __name__ == '__main__':
    cur_dir = os.path.dirname(__file__)
    path = os.path.join(cur_dir, 'data/imgs/')
    files = os.listdir(path)

    output_path = os.path.join(cur_dir, 'data/id/')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    cropper = IDCropper()

    count = 0
    for idx in xrange(len(files)):
        file = files[idx]
        print idx, file

        img = cv2.imread(path + file)
        #Image.fromarray(img).show()
        grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]

        for i in xrange(3):
            gray = grays[-1]
            grays.append(np.rot90(gray))

        for gray in grays:
            try:
                cropped_img = cropper.crop(gray)
            except:
                #traceback.print_exc()
                continue
            if cropped_img is not None:
                break

        else:
            continue
        im = Image.fromarray(cropped_img)
        #im.show()
        im.save(output_path + file)


