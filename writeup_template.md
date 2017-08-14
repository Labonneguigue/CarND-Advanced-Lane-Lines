

# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortion-comparison.jpg "Undistorted"
[corners]: ./output_images/1-undistorted.jpg "corners"
[undist]: ./output_images/straight_lines1-undistorted.jpg "undist"
[sobel]: ./output_images/test6-sobel.png "sobel"
[color]: ./output_images/test6-colorS.png "colorS"
[white]: ./output_images/test6-whiteL.png "whiteL"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[plantuml]: ./processing.png "plantuml"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

Each camera have a particular lens that distort the captured image in a certain way compared to the reality. Since we want to grasp a view of the surroundings as accurately as possible, we need to correct this distortion. OpenCV provides some very useful functions to perform this task.

The code for this step is contained in the `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. To do so I use the `cv2.findChessboardCorners()` function provided by OpenCV.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Here is an example of a successful detection:

![alt text][corners]


I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

I am now going to describe the pipeline each images goes through in order to detect the lines. I am going to display images to illustrate each steps of the way.

#### 1. Undistortion
The pictures of the chessboard were taken with the same camera as the one mounted on the car that took every pictures and videos that were provided for this project. Therefore, after calibrating the camera on the chessboard, we can use the same coefficients to undistort every images and videos of the road.

Since the calibration only needs to happen once, I store the results in a pickle file and load it whenever I need the calibration data.

```python
import calibration as C
mtx, dist = C.LoadCalibrationCoeffs()
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

`LoadCalibrationCoeffs()` can be called from anywhere after importing the calibration file.

Here is the result after undistorting an image. The difference is subtle but we can see that the horizontal line at about y = 600 becomes seems more straight on the right image.

![alt text][undist]


#### 2. Lane detection

I used a combination of color and gradient thresholds to generate a binary image where every non-zeros pixels have a high probability of being part of a lane line.

##### a. Sobel operator

A Sobel operator is an edge-detection algorithm that computed and detect high gradient in a given direction by doing a 2-D convolution over the image. In this case, I chose to detect pixels which luminance (channel L of HLS color space) returns a high gradient in the x direction since the lines I am trying to detect are generally vertical in the image.

![alt text][sobel]

I chose the min and max thresholds for this function to be 20 and 100.

##### b. Color thresholding

Since the lines can be yellow and white, I chose to detect the lane lines by color as well. The S (Sue) channel of the HLS color space can be well suited for extracting a particular color.The following image shows how I extract the yellow lines:

![alt text][color]

I found that extracting only the pixels between 170 and 255 would suit this problem quite well.

To extract the white color, I would chose the L channel and threshold between 200 and 255.

![alt text][white]

We can observe that the white detecting picture detect the right line more accurately.

##### Final result

Here's an example of my output for this step which is an aggregation of these previous steps.  

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

[//]: # (Comment: Way to include images
     m1 : <img:output_images/undistortion-comparison.jpg>)


![alt text][plantuml]

Check if I really want to call the Init() function with an image...
