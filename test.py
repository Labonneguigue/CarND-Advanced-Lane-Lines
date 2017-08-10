import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#from process import *
import detect as D
import calibration as C
import parameters as P

def CalibrationTest(img):
    mtx, dist = P.LoadCalibrationCoeffs()
    dst = cv2.undistort(img, P.mtx, P.dist, None, P.mtx)
    mpimg.imsave(fname[:-4] + '-undistorted.jpg', dst)

def ProcessingPipelineTest(lanesDetector,img, fname):
    result = lanesDetector.ProcessingPipeline(img)
    fname = fname[:-4] + '-processed.png'
    mpimg.imsave(fname, result)
    print("Saved : " + fname)
    return result

def PerspectiveTransformTest(lanesDetector, img, fname, extension=""):
    result = lanesDetector.PerspectiveTransform(img)
    color=[255, 0, 0]
    thickness=1
    cv2.line(result, (D.dst[0][0], D.dst[0][1]), (D.dst[1][0], D.dst[1][1]), color, thickness)
    cv2.line(result, (D.dst[1][0], D.dst[1][1]), (D.dst[2][0], D.dst[2][1]), color, thickness)
    cv2.line(result, (D.dst[2][0], D.dst[2][1]), (D.dst[3][0], D.dst[3][1]), color, thickness)
    cv2.line(result, (D.dst[3][0], D.dst[3][1]), (D.dst[0][0], D.dst[0][1]), color, thickness)
    fname = fname[:-4] + '-PerspectiveTransform' + extension + '.png'
    mpimg.imsave(fname, result)
    print("Saved : " + fname)
    return result

def BlindSlidingWindowsHistogramTest(lanesDetector, img, fname, extension=""):
    lanesDetector.BlindSlidingWindowsHistogram(img)
    result, ploty, l_fit, r_fit = lanesDetector.VisualizePolynomial()
    #plt.subplot((1,1,1))
    plt.close('all')
    plt.imshow(result)
    plt.plot(l_fit, ploty, color='yellow')
    plt.plot(r_fit, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    fname = fname[:-4] + '-histogram' + extension + '.png'
    plt.savefig(fname)
    print("Saved : " + fname)
    #mpimg.imsave(fname[:-4] + '-histogram' + extension + '.jpg', result)

def DisplayParameters():
    print(P.parameters['orig_points_x'])
    print(P.parameters['orig_points_x'][0])
    print(P.parameters['orig_points_x'][1])
    print(P.parameters['orig_points_x'][2])
    print(P.parameters['orig_points_x'][3])
    print(P.parameters['orig_points_y'])

if __name__ == "__main__":
    #fname = 'test_images/straight_lines1.jpg'
    #fname = 'test_images/straight_lines2.jpg'
    #fname = 'test_images/test1.jpg'
    fname = 'test_images/test2.jpg'
    print("Test performed on : " + fname)
    output_dir = "test_images/"
    img = D.LoadImage(fname)
    D.Init(img)
    print(img.shape)
    if 0:
        CalibrationTest()
    if 0:
        ProcessingPipelineTest(img)
    if 0:
        DisplayParameters()
    if 0:
        img = cv2.undistort(img, D.mtx, D.dist, None, D.mtx)
        PerspectiveTransformTest(img, fname, "-undistorted")
    if 1:
        lanesDetector = D.LanesDetector()
        dir = os.listdir("test_images/")
        for file in dir:
            if (file != '.DS_Store') & (file[-4:] == '.jpg'):
                print(file)
                img = D.LoadImage("test_images/" + file)
                img = ProcessingPipelineTest(lanesDetector, img, output_dir+file)
                img = PerspectiveTransformTest(lanesDetector, img, output_dir+file)
                BlindSlidingWindowsHistogramTest(lanesDetector, img, output_dir+file)
                print(lanesDetector.leftPolynomialFit)
                print(lanesDetector.rightPolynomialFit)
                print()
