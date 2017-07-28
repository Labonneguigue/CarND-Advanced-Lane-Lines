import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#from process import *
import process as P
from calibration import *
from parameters import *

def CalibrationTest(img):
    mtx, dist = P.LoadCalibrationCoeffs()
    dst = cv2.undistort(img, P.mtx, P.dist, None, P.mtx)
    mpimg.imsave(fname[:-4] + '-undistorted.jpg', dst)

def ProcessingPipelineTest(img):
    result = P.ProcessingPipeline(img)
    mpimg.imsave(fname[:-4] + '-processed.jpg', result)
    return result

def PerspectiveTransformTest(img):
    result = P.PerspectiveTransform(img)
    mpimg.imsave(fname[:-4] + '-PerspectiveTransform.jpg', result)

def DisplayParameters():
    print(parameters['orig_points_x'])
    print(parameters['orig_points_x'][0])
    print(parameters['orig_points_x'][1])
    print(parameters['orig_points_x'][2])
    print(parameters['orig_points_x'][3])
    print(parameters['orig_points_y'])

if __name__ == "__main__":
    fname = 'test_images/straight_lines1.jpg'
    #fname = 'test_images/test1.jpg'
    #fname = 'test_images/test2.jpg'
    img = P.LoadImage(fname)
    P.Init(img)
    print(img.shape)
    if 0:
        CalibrationTest()
    if 0:
        ProcessingPipelineTest(img)
    if 0:
        DisplayParameters()
    if 1:
        #img = cv2.undistort(img, P.mtx, P.dist, None, P.mtx)
        PerspectiveTransformTest(img)
    if 0:
        img = cv2.undistort(img, P.mtx, P.dist, None, P.mtx)
        img = ProcessingPipelineTest(img)
        PerspectiveTransformTest(img)
