# Project Advanced Lane Finding

# necessary imports
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
from moviepy.editor import VideoFileClip

# change working dir to path containing this file
os.chdir(os.path.dirname('C:/Users/P325748/Documents/4_AI/Udacity/CarND-Advanced-Lane-Lines-master/'))

# class line for tracking left and right line/lane
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #low-pass filter constant for polynomial fitting
        self.n_lowpass = 3
        #polynomial coefficients averaged over the last n iterations
        #self.best_fit = None  
        self.best_fit = np.ndarray(shape=(self.n_lowpass,3), dtype=float)
        self.best_fit.fill(np.nan)
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        #was detection (0) or tracking (1) executed?
        self.fitting_algo = [0]     # firs iteration will start with line detection

# step 1: find camera coefficients -- calibration
# create camera and distortion coefficients
# function "calc_calibration_coeff"
# input: image-folder img_folder which contains images for calibration, numbers of edges in x and y for calibration board nx and ny
# returns camera matric coefficients mtx (pickles), camera distance coeef dist
def calc_calibration_coeff(img_folder, nx, ny):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    ###obj_pts = np.zeros((6*7,3), np.float32) # for (7,6) chessboard
    ###obj_pts[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) # for (7,6) chessboard
    obj_pts = np.zeros((nx*ny,3), np.float32) # for (nx, ny) chessboard
    obj_pts[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # for (nx, ny) chessboard
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    #os.listdir(img_folder + '*.jpg')
    #images = glob.glob(img_folder + '*.jpg')    
    images = glob.glob('./camera_cal/' + '*.jpg')    
    for fname in images:
        print('Calibrate with ', fname)
        img = cv2.imread(fname)                    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print('Found chessboard.')
            objpoints.append(obj_pts)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners2, ret)    
            #plt.figure()
            #plt.imshow(img)                                       
    
    #plt.show()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # test calibration with first calib-image (only for debug) --> calibration works
    img = cv2.imread(images[0])    
    warped = cv2.undistort(img, mtx, dist)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.imshow(warped)
    plt.savefig('./output_images/' + images[0][13:-4] + '_cal.jpg')
    plt.close()
    #plt.show()

    # return distortion and coeff-matrix
    return mtx, dist

def warp_undistort(img, mtx_cal, dist_cal): 
    # undistort image with calibration data
    undist = cv2.undistort(img, mtx_cal, dist_cal)
    # corresponding coordinates determined with example straight image
    src = np.float32([[200,720],[1100,720],[685,450],[590,450]])
    dst = np.float32([[320,720],[950,720],[950,0],[320,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0])) 

    return warped, undist, M, Minv

def apply_HLS_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channe
    img = cv2.medianBlur(img,11)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    r_channel = img[:,:,0]

    # idea taken from https://chatbotslife.com/self-driving-cars-advanced-computer-vision-with-opencv-finding-lane-lines-488a411b2c3d
    # filter ranges of colors yellow and white
    #yellow = cv2.inRange(hsv, (90, 100, 100), (110, 255, 255))        
    yellow = cv2.inRange(hsv, (90, 60, 100), (110, 255, 255))
    white = cv2.inRange(img, (180,180,180), (255,255,255))    

    combined_color_binary = np.zeros_like(yellow)          # thresholded color images    
    combined_color_binary[(yellow == 255) | (white == 255)] = 1

    # Sobel x of s-channel
    #sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobelx = cv2.Sobel(combined_color_binary, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # combine color and sobel
    combined_color_sobel_binary = np.zeros_like(yellow)          # thresholded color images    
    combined_color_sobel_binary[(yellow == 255) | (white == 255) | (sxbinary == 1)] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255    
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, combined_color_binary)) * 255    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, combined_color_binary)) * 255
    
    # combine both sobel-gradient hls and color-channel
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    combined_binary[(combined_color_binary == 1) | (sxbinary == 1)] = 1    

    # set final output only to combined_color_binary with checks for yellow and white
    combined_binary = combined_color_binary
    #combined_binary = sxbinary      # new version: sobel of color-filter
    color_binary = combined_binary
    #combined_binary = combined_color_sobel_binary

    # debug for setting up threshold parameters
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.subplot(2,1,2)
    # plt.imshow(color_binary)
    ##plt.imshow(cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB))
    #plt.imshow(r_channel)
    #plt.subplot(2,2,3)
    #plt.imshow(s_channel)
    #plt.subplot(2,2,4)
    #plt.imshow(color_binary)
    #plt.show()

    return combined_binary, color_binary

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin; old value = 100
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### Find the four below boundaries of the window ###
        win_xleft_low = np.max([leftx_current - margin, 0]) # win_xleft = [max_peak_left - margin --> max_peak_left + margin]
        win_xleft_high = np.min([leftx_current + margin, binary_warped.shape[1]-1])  # 
        win_xright_low = np.max([rightx_current - margin, 0])  # win_xright = [max_peak_right - margin --> max_peak_right + margin]
        win_xright_high = np.min([rightx_current + margin, binary_warped.shape[1]-1])  # 
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        # identify nonzero-pixels indices instead of x and y
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        
        # Append these indices to the lists
        #left_lane_inds.append(good_left_inds.tolist()) 
        #right_lane_inds.append(good_right_inds.tolist())  
        left_lane_inds = left_lane_inds + good_left_inds.tolist()
        right_lane_inds = right_lane_inds + good_right_inds.tolist()

        # indices should be used for index of nonzerox --> 
        # relevant data: 
        # binary_warped     [720 x 1280]    == 921.600
        # nonzerox          [0:89486]       == subset of binary_warped
        # window            [80 x 200]      == 1.600

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix: 
            leftx_current = np.mean(nonzerox[good_left_inds]).astype(int)

        if len(good_right_inds) > minpix:
            rightx_current = np.mean(nonzerox[good_right_inds]).astype(int)
                
        # Concatenate the arrays of indices (previously was a list of lists of pixels)    
        #left_lane_inds = np.concatenate(left_lane_inds)
        #right_lane_inds = np.concatenate(right_lane_inds)    

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, histogram

def fit_polynomial(out_img):
    # Find our lane pixels first
    #left_line.allx, left_line.ally, right_line.allx, right_line.ally, out_img, histogram = find_lane_pixels(binary_warped)
    #leftx, lefty, rightx, righty, out_img, histogram = find_lane_pixels(binary_warped)

    # line detected / detectable, if more than 5000 points for each line were detected
    MIN_PIXELS_FOR_LINE = 500
    if (len(left_line.allx) > MIN_PIXELS_FOR_LINE): 
        left_line.detected = True
    else:
        left_line.detected = False

    if (len(right_line.allx) > MIN_PIXELS_FOR_LINE): 
        right_line.detected = True
    else:
        right_line.detected = False

    if (left_line.detected == True) & (right_line.detected == True): 
        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        left_fit = np.polyfit(left_line.ally, left_line.allx, 2)
        right_fit = np.polyfit(right_line.ally, right_line.allx, 2)

        # calculate difference to last iteration
        left_line.diffs = left_fit - left_line.current_fit
        right_line.diffs = right_fit - right_line.current_fit

        # step 4a: low-pass output from polynomial --> final polynomial is averaged on last n frames
        if (left_line.detected == False) | (right_line.detected == False) | (np.sum(np.abs(left_line.diffs)) > 40):
            left_line.fitting_algo.append(0)
        else:
            left_line.fitting_algo.append(1)

        # and store new value to line-objects
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # low pass filter by setting left_fit to be a mean of last n values
        # implemented ring buffer --> rotate by 1x first
        left_line.best_fit = np.roll(left_line.best_fit, -1, axis=0)
        right_line.best_fit = np.roll(right_line.best_fit, -1, axis=0)
        left_line.best_fit[left_line.n_lowpass-1] = left_fit
        right_line.best_fit[left_line.n_lowpass-1] = right_fit

        # build average of best_fit for image output
        left_fit = np.nanmean(left_line.best_fit,axis=0)
        right_fit = np.nanmean(right_line.best_fit,axis=0)

        # Generate x and y values for plotting and final result --> only here our average is used
        ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            #left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty + left_fit[3]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            #right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty + right_fit[3]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[left_line.ally, left_line.allx] = [255, 0, 0]
        out_img[right_line.ally, right_line.allx] = [0, 0, 255]
    else:
        # if no lines were found
        left_fitx = right_fitx = ploty = 0
    
    return left_fitx, right_fitx, ploty, out_img

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###    
    left_lane_inds = (nonzerox > (left_line.current_fit[0]*nonzeroy**2 + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] 
    - margin)) & (nonzerox < (left_line.current_fit[0]*nonzeroy**2 + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin))
    right_lane_inds = (nonzerox > (right_line.current_fit[0]*nonzeroy**2 + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] 
    - margin)) & (nonzerox < (right_line.current_fit[0]*nonzeroy**2 + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255    

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]    
    
    return leftx, lefty, rightx, righty, out_img    

def measure_curvature_real(left_fit_cr, right_fit_cr, ploty, img_width):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''

    if((left_line.detected == True) & (right_line.detected == True)):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
            
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        
        # curvature without conversion pixel -> meters
        #left_curverad = ((1 + (2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2)) / (np.abs(2*left_fit[0]))  ## Implement the calculation of the left line here
        #right_curverad = ((1 + (2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2)) / (np.abs(2*right_fit[0]))  ## Implement the calculation of the right line here
            
        # Scaling of parabola: x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**(3/2)) / (np.abs(2*right_fit_cr[0]))  ## Implement the calculation of the right line here    
        
        # calculate deviation by.. 
        # 1) subtracting difference of x-values of right and left polynomial at lowest image point ([-1], highest y-coord.) from origin (width/2)
        # 2) multiplying by xm-factor
        dev = ((right_line.recent_xfitted[-1] - left_line.recent_xfitted[-1]) - img_width/2) * xm_per_pix
    else:
        # if no line was found
        left_curverad = right_curverad = dev = 0

    return left_curverad, right_curverad, dev

def find_lines(img):  
    if BGR_MODE == True: 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if debug:         
        ax = plt.figure()    
        plt.subplot()
        ax = plt.subplot(3,2,1)
        ax.set_title('Original')    
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # step 2: undistort image with calibration data and warp image
    warped, undist, M, Minv = warp_undistort(img, mtx_cal, dist_cal)    

    if debug: 
        ax = plt.subplot(3,2,2)
        ax.set_title('Warp and undistort')
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    # step 3: convert to HLS and apply threshold                   
    binary_warped, color_warped = apply_HLS_threshold(warped)          
    if debug:
        ax = plt.subplot(3,2,3)
        ax.set_title('HLS and binary thresholded')
        plt.imshow(color_warped)

    # step 4a: find lane pixels, if no line was detected yet and distance to previous measurement too high
    if left_line.fitting_algo[-1] == 0:     # if distance of previous measurements too high, fitting_algo is set to 0
        left_line.allx, left_line.ally, right_line.allx, right_line.ally, img_lane_pixel, hist = find_lane_pixels(binary_warped)        
    else:
        left_line.allx, left_line.ally, right_line.allx, right_line.ally, img_lane_pixel = search_around_poly(binary_warped)        
        hist = []       # no histogram in case of line tracking

    # step 4: find lane pixels and fit polynomial
    left_line.recent_xfitted, right_line.recent_xfitted, ploty, out_img = fit_polynomial(img_lane_pixel)        
    
    # step 5: calculate curvature from polynomial
    left_line.radius_of_curvature, right_line.radius_of_curvature, dev = measure_curvature_real(left_line.current_fit, right_line.current_fit, ploty, img.shape[1])
    #left_curverad, right_curverad = measure_curvature_real(left_fitx, right_fitx, ploty)
    if debug:
        ax = plt.subplot(3,2,4)    
        ax.set_title('Sliding Window Histogram')
        plt.plot(hist)    
        ax = plt.subplot(3,2,5)    
        ax.set_title('Fitted polynomial')
        plt.imshow(out_img)
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_line.recent_xfitted, ploty, color='yellow')
        plt.plot(right_line.recent_xfitted, ploty, color='yellow')
        # plot warped and undistorted image with lines and curvature    
        ax = plt.subplot(3,2,6)    
        ax.set_title('Curve: ' + repr(left_line.radius_of_curvature) + ', ' + repr(right_line.radius_of_curvature))    
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.plot(left_line.recent_xfitted, ploty, color='yellow')
        plt.plot(right_line.recent_xfitted, ploty, color='yellow')    

    # step 6: visualize lanes on original images
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.recent_xfitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.recent_xfitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # add text for curvature and position in lane
    text = 'Curv.: %.2f, %.2f, Dev.: %.2f' % (left_line.radius_of_curvature,right_line.radius_of_curvature, dev)
    result = cv2.putText(result, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)    

    if BGR_MODE == True: 
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    if debug:
        plt.figure()
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))        

    return result

# step 1: execute camera calibration
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y
mtx_cal, dist_cal = calc_calibration_coeff('./camera_cal/', nx, ny)

# load images and prepare
#images = glob.glob('./test_images2/' + '*.jpg')    # test-images2 : BGR_MODE = False
images = glob.glob('./test_images/' + '*.jpg')    # test-images : BGR_MODE = True

# right and left line
left_line = Line()
right_line = Line()

# images used are in BGR-Mode, videos in RGB
BGR_MODE = False

# debug mode to create images for documentation / debugging
debug = True

## run lane finding on images
for fname in images:
    print('Working on ', fname)
    img = cv2.imread(fname)      
    result = find_lines(img)    
    plt.savefig('./output_images/' + fname[14:-4] + '_std.jpg')   # save finale result to new file  
    plt.close()      
    plt.savefig('./output_images/' + fname[14:-4] + '_detail.jpg')   # save detailled result to new file  
    plt.close()
    left_line.detected = False          # in order to avoid calling line-tracking instead of detection
    left_line.best_fit.fill(np.nan)          # and reset best_fit history for single images
    right_line.best_fit.fill(np.nan)          # and reset best_fit history for single images
    left_line.fitting_algo[-1] = 0          
    print('Difference to prev. frame: Left: ', left_line.diffs, '; Right: ', right_line.diffs)
    print('Abs. Difference Sum: ', np.sum(np.abs(left_line.diffs)))    
    print('Done.')

    # show warping steps and warping matrix
    if debug: 
        ax = plt.figure()
        if BGR_MODE:
            warp_method = img          
        else:            
            warp_method = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            
        warped, unwarped, M, Minv = warp_undistort(warp_method, mtx_cal, dist_cal)
        #src = np.float32([[200,720],[1100,720],[685,450],[590,450]])
        #dst = np.float32([[320,720],[950,720],[950,0],[320,0]])
        unwarped = cv2.line(unwarped, (200,720), (1100,720), (255,0,0),3)
        unwarped = cv2.line(unwarped, (1100,720), (685,450), (255,0,0),3)
        unwarped = cv2.line(unwarped, (685,450), (590,450), (255,0,0),3)
        unwarped = cv2.line(unwarped, (590,450), (200,720), (255,0,0),3)
        warped = cv2.line(warped, (320,720), (950,720), (255,0,0),3)
        warped = cv2.line(warped, (950,720), (950,0), (255,0,0),3)
        warped = cv2.line(warped, (950,0), (320,0), (255,0,0),3)
        warped = cv2.line(warped, (320,0), (320,720), (255,0,0),3)

        #if BGR_MODE:
        #    unwarped = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)   
        #    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)   

        plt.figure()
        plt.subplot(1,2,1)        
        plt.imshow(unwarped)
        plt.subplot(1,2,2)
        plt.imshow(warped)
        plt.savefig('./output_images/' + fname[14:-4] + '_warp.jpg')

# load video files
# challenge video
BGR_MODE = True

FPS = 0.1 # debug: grab frame every FPS-seconds 
t_start = 0



# Problems in project_video.mp4:
# @ 20 - 25 s
# @ 37 - 42 s
# for nframe in range(50):
#     img = clip1.get_frame(t_start + nframe * FPS)
#     print('Working on frame ', nframe)         
#     result = find_lines(img)    
#     plt.savefig('./output_videos/' + t_start.__str__() + '_' + nframe.__str__() + '_std.jpg')   # save final result to new file  
#     plt.close()      
#     plt.savefig('./output_videos/' + t_start.__str__() + '_' + nframe.__str__() + '_detail.jpg')   # save detailled result to new file  
#     plt.close()    
#     print('Difference to prev. frame: Left: ', left_line.diffs, '; Right: ', right_line.diffs)
#     print('Abs. Difference Sum: ', np.sum(np.abs(left_line.diffs)))    
#     print('Done.')

# project video
clip2 = VideoFileClip("project_video.mp4")
white_output2 = './output_videos/project_video_std.mp4'

white_clip2 = clip2.fl_image(find_lines)
white_clip2.write_videofile(white_output2, audio=False)

white_output3 = './output_videos/harder_challenge_video_std.mp4'
clip3 = VideoFileClip("harder_challenge_video.mp4")
white_clip3 = clip3.fl_image(find_lines)
white_clip3.write_videofile(white_output3, audio=False)

white_output1 = './output_videos/challenge_video_std.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
white_clip1 = clip1.fl_image(find_lines)
white_clip1.write_videofile(white_output1, audio=False)

print('Done.')