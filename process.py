import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calibration import *
from parameters import *

'''
This variable should be set to true when the calibration
data has been loaded and the preprocessing pipeline can
be executed
'''
initialized = False
'''
Global variables
'''
mtx = None
dist = None
PersMat = None

def LoadImage(fname = 'test_images/test5.jpg'):
    '''
    Load and returns an image - RGB color space
    '''
    return mpimg.imread(fname)

def ColorChannelBinarization(s_channel, s_thresh=(170, 255)):
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary

def SobelBinarization(l_channel, s_thresh=(20, 100), direction='x'):
    '''
    Performs Sobel derivative function in either x or y direction
    Expects an image L channel from its HLS representation.
    '''
    if direction == 'x':
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    else:
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    abs_sobel = np.absolute(sobel) # Absolute derivative to accentuate lines away from horizontal/vertical
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Threshold x gradient
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= s_thresh[0]) & (scaled_sobel <= s_thresh[1])] = 1
    return sbinary


def StackImages(imgA, imgB, imgC=None):
    '''
    Stack 3 2D images into a 3D image.
    The 2 images need to have the same shape.
    '''
    assert(imgA.shape == imgB.shape)
    if imgC==None:
        return np.dstack((np.zeros_like(imgA), imgA, imgB))
    else:
        return np.dstack((imgA, imgB, imgC))


def DisplayAndSave2Images(imgA, imgB, name):
    '''
    Display 2 images side by side and save the figure
    for a quick comparison
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(imgA)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(imgB)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('output_images/' + name)

def InitPerspectiveMatrix(img):
    '''
    Order of the points
            1 ---- 2        <- parameters['orig_points_y']
           /        \
          /          \
         /            \
        4 ------------ 3
    '''
    src = np.float32(
        [[parameters['orig_points_x'][0], parameters['orig_points_y']],
        [parameters['orig_points_x'][1], parameters['orig_points_y']],
        [parameters['orig_points_x'][2], img.shape[0]],
        [parameters['orig_points_x'][3], img.shape[0]]])
    dst = np.float32(
        [[(img.shape[1] / 4), 0],
        [(img.shape[1] * 3 / 4), 0],
        [(img.shape[1] * 3 / 4), img.shape[0]],
        [(img.shape[1] / 4), img.shape[0]]])
    print(src)
    print(dst)
    return cv2.getPerspectiveTransform(src, dst)

def PerspectiveTransform(img):
    '''
    Change the perspective of the image to obtain
    a top down view of the road
    '''
    return cv2.warpPerspective(img, PersMat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def Init(img):
    '''
    Initialization function to set every global variables
    used for computation later on.
    '''
    global mtx
    global dist
    global initialized
    mtx, dist = LoadCalibrationCoeffs()
    assert(mtx != None)
    assert(dist != None)

    global PersMat
    PersMat = InitPerspectiveMatrix(img)
    assert(PersMat != None)

    initialized = True


def ProcessingPipeline(img):
    '''
    This processing pipeline is composed of the following steps:
     - Image undistortion
     - Image Binarization
     - Perspective Transform
    This needs to be applied to every frame before starting detecting the lane lines
    '''
    assert(initialized)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    SobelBinary = SobelBinarization(l_channel)  # sobel along x
    ColorBinary = ColorChannelBinarization(s_channel)

    return StackImages(SobelBinary, ColorBinary)


if __name__ == "__main__":
    img = LoadImage()
    Init(img)
