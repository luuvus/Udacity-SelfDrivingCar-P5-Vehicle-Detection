# Vehicle Detection and Tracking

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_visual.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_windows_2.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/car_positions_and_heat.png
[video1]: ./project_video.mp4
[img_bboxes_and_heat1]: ./examples/bboxes_and_heat1.png
[img_bboxes_and_heat2]: ./examples/bboxes_and_heat2.png
[img_bboxes_and_heat3]: ./examples/bboxes_and_heat3.png
[img_bboxes_and_heat4]: ./examples/bboxes_and_heat4.png
[img_bboxes_and_heat5]: ./examples/bboxes_and_heat5.png
[img_bboxes_and_heat6]: ./examples/bboxes_and_heat6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
I will use the rubric points individually and I addressed each point in my implementation.  

---

## Histogram of Oriented Gradients (HOG)

**1. Explain how (and identify where in your code) you extracted HOG features from the training images.**

The code for this step is contained in the eight code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

**2. Explain how you settled on your final choice of HOG parameters.**

I started off by expermenting with the parameters for the HOG function (orientations, pixels_per_cell, and cells_per_block) and then applied color space well until highest accuracy is achieved. In the end,  I applied three channels with YCrCb color space and HOG parameters of orientations=9, pixels_per_cell=(8, 8) and cells_per_block=(2, 2). This combination usually yield the best test accuracy on the linear SVM model.

**3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**

Detail training implementation is on cell 15 of IPython notebook. 

The linear SVM is trained using combination of the spatial features, histogram features, and HOG features. The training images were converted from RGB color space to YCrCb color space before extracting the features. All feature values are calculated with a distribution in the range from 0 to 1 to avoid any feature becoming dominant. 


## Sliding Window Search

**1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?**

I search the region between y=[400,700] since this is the only region where cars shoud appear. I use window size 64x64 and overlap 50%, resize it to 64*64 and feed it to the classifier. Here is the result:

![alt text][image3]

**2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?**

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

![alt text][image4]
---

## Video Implementation

**1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)**

Here's a [link to my video result](./videos_output/project_video.mp4)


**2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][img_bboxes_and_heat1]
![alt text][img_bboxes_and_heat2]
![alt text][img_bboxes_and_heat3]
![alt text][img_bboxes_and_heat4]
![alt text][img_bboxes_and_heat5]
![alt text][img_bboxes_and_heat6]

### Here the resulting bounding boxes and output of `scipy.ndimage.measurements.label()` on the integrated heatmap are drawn onto the last frame in the series:
![alt text][image7]



---

## Discussion

**1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**

The objects detection pipeline work well, but the process requires a lot of manual parameters tuning, which takes a lot of time and cannot be automatically adapt to a new video with different driving environment. In addition, the current pipeline cannot detect other vehicle types, such as 18-wheeler trucks and motocyles. False positives need further improvement to fully elimiate false detection at certain frames. 

The processing speed is very slow and cannot be used on live video feed/stream. One of the way to improve processing speed is to try Convolution Neural Network(CNN) as classifier instead of HOG classifier even thought model training time on CNN will take longer.