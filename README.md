# CarND-Advanced-Lane-Lines
Part of Nanodegree program 

## Camera Matrix and Distortion Coefficients
The camera calibration was done with the given images from a chessboard with a 9x6 pattern. The pattern could be found in 17 of 20 images. The figure below shows a distorted (top) and an undistorted chessboard pattern image. In order to calibrate an image, all detected points in the image are mapped to 3d object space. The mapping between those spaces is calculated within "calibrateCamera"-function.
The camera calibration is implemented in "calc_calibration_coeff".

![](calibration1_cal.jpg)

## Pipeline for advanced lane finding
The pipeline contains following major steps which are shortly described in seperate chapters: 
	1. Perspective Transform
	2. Color Transform and gradients
	3. Identification of lane line pixels
	4. Polynomial fitting of curvature
	5. Calculation of Curvature and position of vehicle in lane
Final image with lane area (back-warping)

### Perspective Transform
The perspective transform is needed in order to generate a top-view-image s.t. the curvature of the street can be measured. To find corresponding locations, a trapezoid was fitted on an image with straigt lines. These points were used as the mapping function between the "undistorted, undwarped" and "undistorted, warped" image. The mapping matrix is given below: 
Original x	| Original y |	New x |	New y	 |-----
------------|------------|--------|--------|-----
200	| 720	| 320 |	720 |	Bottom left
1150 | 720	| 950	| 720 |	Bottom right
700	| 450	| 950 |	0	| Top right
600 |	450	| 320 |	0	| Top left

The output of the perspective transform can be seen in the image below: 

![](calibration1_cal.jpg)
### 
