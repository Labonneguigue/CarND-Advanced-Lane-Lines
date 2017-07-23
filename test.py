import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from process import *
from calibration import *
from parameters import *

def CalibrationTest(fname = 'test_images/test1.jpg'):
    img = LoadImage(fname)
    mtx, dist = LoadCalibrationCoeffs()
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    mpimg.imsave(fname[:-4] + '-undistorted.jpg', dst)

if __name__ == "__main__":

    if 0:
        CalibrationTest()
    if 0:
        img = LoadImage()
        res = Binarization(img)
        DisplayAndSave2Images(img, res, 'binarization.jpg')

    print(parameters['orig_points_x'])
    print(parameters['orig_points_x'][0])
    print(parameters['orig_points_x'][1])
    print(parameters['orig_points_x'][2])
    print(parameters['orig_points_x'][3])
    print(parameters['orig_points_y'])
