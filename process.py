import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calibration import *

'''
This variable should be set to true when the calibration
data has been loaded and the preprocessing pipeline can
be executed
'''
initialized = False
mtx = None
dist = None

def LoadImage(fname = 'test_images/test5.jpg'):
    '''
    Load and returns an image - RGB color space
    '''
    return mpimg.imread(fname)

def Binarization(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

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

def PerspectiveTransform():
    

def Init():
    global mtx
    global dist
    global initialized
    mtx, dist = LoadCalibrationCoeffs()
    assert(mtx != None)
    assert(dist != None)
    initialized = True

def PreprocessingPipeline(img):
    '''
    This preprocessing pipeline is composed of the following steps:
     - Image undistortion
     - Image Binarization
     - Perspective Transform
    This needs to be applied to every frame before starting detecting the lane lines
    '''
    assert(initialized)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    binarized = Binarization(img)
