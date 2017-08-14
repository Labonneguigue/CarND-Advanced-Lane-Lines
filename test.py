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

def showimg(subplace, title, _img):
    plt.subplot(*subplace)
    plt.axis('off')
    plt.title(title)
    plt.imshow(_img)
    plt.tight_layout()

def CalibrationTest(img, fname):
    mtx, dist = C.LoadCalibrationCoeffs()
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig('output_images/' + os.path.basename(fname)[:-4] + '-undistorted.jpg')
    #mpimg.imsave(fname[:-4] + '-undistorted.jpg', dst)

def ProcessingPipelineTest(lanesDetector,img, fname):
    result = lanesDetector.ProcessingPipeline(img)
    fname = 'output_images/' + os.path.basename(fname)[:-4] + '-processed.png'
    mpimg.imsave(fname, result, cmap=plt.cm.gray)
    print("Saved : " + fname)
    return result

def SobelTest(img, fname):
    mtx, dist = C.LoadCalibrationCoeffs()
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    SobelBinary = D.SobelBinarization(l_channel)
    fname = 'output_images/' + os.path.basename(fname)[:-4] + '-sobel.png'
    mpimg.imsave(fname, SobelBinary, cmap=plt.cm.gray)
    print("Saved : " + fname)

def ColorTest(img, fname):
    mtx, dist = C.LoadCalibrationCoeffs()
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,2]
    ColorBinarized = D.ColorChannelBinarization(s_channel)
    fname = 'output_images/' + os.path.basename(fname)[:-4] + '-colorS.png'
    mpimg.imsave(fname, ColorBinarized, cmap=plt.cm.gray)
    print("Saved : " + fname)

def WhiteColorTest(img, fname):
    mtx, dist = C.LoadCalibrationCoeffs()
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    WhiteBinarized = D.ColorChannelBinarization(l_channel, (200, 255))
    fname = 'output_images/' + os.path.basename(fname)[:-4] + '-whiteL.png'
    mpimg.imsave(fname, WhiteBinarized, cmap=plt.cm.gray)
    print("Saved : " + fname)

def PerspectiveTransformTest(lanesDetector, img, fname, extension=""):
    result = lanesDetector.PerspectiveTransform(img, D.PersMat)
    color=[255, 0, 0]
    thickness=2
    cv2.line(result, (D.dst[0][0], D.dst[0][1]), (D.dst[1][0], D.dst[1][1]), color, thickness)
    cv2.line(result, (D.dst[1][0], D.dst[1][1]), (D.dst[2][0], D.dst[2][1]), color, thickness)
    cv2.line(result, (D.dst[2][0], D.dst[2][1]), (D.dst[3][0], D.dst[3][1]), color, thickness)
    cv2.line(result, (D.dst[3][0], D.dst[3][1]), (D.dst[0][0], D.dst[0][1]), color, thickness)

    cv2.line(img, (D.src[0][0], D.src[0][1]), (D.src[1][0], D.src[1][1]), color, thickness)
    cv2.line(img, (D.src[1][0], D.src[1][1]), (D.src[2][0], D.src[2][1]), color, thickness)
    cv2.line(img, (D.src[2][0], D.src[2][1]), (D.src[3][0], D.src[3][1]), color, thickness)
    cv2.line(img, (D.src[3][0], D.src[3][1]), (D.src[0][0], D.src[0][1]), color, thickness)
    fname = os.path.basename(fname)[:-4] + '-persp' + extension + '.png'
    D.DisplayAndSave2Images(img, result, fname, "Perspective Transform", True)
    #mpimg.imsave(fname, result)
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
    #fname = 'test_images/test2.jpg'
    fname = 'test_images/test6.jpg'
    print("Test performed on : " + fname)
    output_dir = "test_images/"
    img = D.LoadImage(fname)
    D.Init(img)
    img = cv2.undistort(img, D.mtx, D.dist, None, D.mtx)
    if 0:
        CalibrationTest(img, fname)
    if 0:
        result = ProcessingPipelineTest(D.LanesDetector(), img, fname)
        D.DisplayAndSave2Images(img, result, os.path.basename(fname)[:-4] + "-side.png", True)
    if 0:
        SobelTest(img, fname)
    if 0:
        ColorTest(img, fname)
    if 0:
        WhiteColorTest(img, fname)
    if 0:
        DisplayParameters()
    if 1:
        PerspectiveTransformTest(D.LanesDetector(), img, fname)
    if 0:
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
