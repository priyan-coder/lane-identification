from cv2 import cv2
import numpy as np
# import matplotlib.pyplot as plt


# grays, reduces noise and returns an image with
# detected edges based on gradient between adjacent pixels
# canny is a 2-dimensional array
def cannyAlgo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# returns the enclosed region of field of view
def regionOfInterest(img):
    heightOfImage = img.shape[0]
    polygons = np.array([
        [(200, heightOfImage), (1100, heightOfImage), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, [255, 255, 255])
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# to generate coordinates for the lines we need to draw
def make_coordinates(img, line_params):
    # unpack the array of parameters
    slope, intercept = line_params

    # we hardcode y1 = height of image
    # because we want our line to start
    # at the bottom of the image
    y1 = img.shape[0]

    # we hardcode y2 to be 3/5  of the height of the image
    y2 = int(y1 * (3/5))

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def averageSlopeIntercept(img, linesDetected):
    # slope and intercept pairs of the line displayed on the left of image
    left_fit = []
    # slope and intercept pairs of the line displayed on the right of image
    right_fit = []

    for line in linesDetected:
        # unpack coordinates
        x1, y1, x2, y2 = line.reshape(4)
        # capture the slope (m) and y-intercept(b) for each line
        # polyfit a degree 1 eqn ( y = mx + b_)
        # and returns a vector of coefficients
        # that describe the eqn
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # average out the values into a single slope and y-intercept
    # i.e. a single line for left and right side respectively
    # the axis is set to 0
    # to indicate that we got to work vertically down the rows
    # [(slope1, intercept1), (slope2, intercept2)]
    # to get the average slope and intercept values

    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)

        left_line = make_coordinates(img, left_fit_average)
        right_line = make_coordinates(img, right_fit_average)

        return np.array([left_line, right_line])


def displayLines(img, linesDetectedInGradientImage):
    blackImgThatFollowsShapeOfimg = np.zeros_like(img)

    # if lines are detected
    if linesDetectedInGradientImage is not None:
        # each line is a 1d array
        for x1, y1, x2, y2 in linesDetectedInGradientImage:
            # draw this line that join the coordinates
            # on the black image which traces the parent image
            # with the BGR values (255,0,0)
            # and 10 px thickness
            cv2.line(blackImgThatFollowsShapeOfimg,
                     (x1, y1), (x2, y2), (255, 0, 0), 10)

    return blackImgThatFollowsShapeOfimg


# VIDEO PROCESSING
# video capture object
cap = cv2.VideoCapture("test2.mp4")

# returns true if videoCapture is initialized
while(cap.isOpened()):
    # decode every video frame
    # first value is a boolean which doesnt interest us
    # secodn value is the current frame where we detect lines
    # reuse algo for image line detection and replace
    # lane_image with current frame
    _, current_frame = cap.read()
    canny_image = cannyAlgo(current_frame)

    cropped_image = regionOfInterest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_Lines = averageSlopeIntercept(current_frame, lines)
    line_image = displayLines(current_frame, averaged_Lines)
    comboImage = cv2.addWeighted(current_frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", comboImage)

    # we want to wait 1 ms between frames
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


# ---------------------THE LOGIC BEHIND THE CODE------------------

# conda install -c conda-forge opencv
# check pip installation online

# EDGE DETECTION = sharp change of intensity b/w adjacent pixels
# sharp change = steep gradient
# width or x axis ==> number of cols in an image
# height or y axis ==> number of rows in an image
# which increases downwards i.e. top of image is 0 )
# The image can be viewed as a function of x and y ==> f(x,y)
# The derivative ==> gradient


# intensities = color
# ranges from 0 ( black ) to 255( white )
# image is an array
# each row is am array of pixel values

# 1. Convert image to grayscale

# 2. Reduce noise using GaussianBlur
# Each pixel value is set to the
# weighted average of the surrouding pixels
# using a 5x5 kernel with zero deviation

# 3. Find derivative image ==> Canny
# low threshold pixels rejected
# high threshold pixels accepted as edge pixel
# those between low and high will be accepted only if
# that pixel is connected to a edge pixel
# use a low - high threshold of 1 : 3
# Gradients that exceed the high threshold
# as traced as bright pixels in white
# Gradients that are small and below the low threshold are black


# 4. Prepare a mask to set the region of interest
# shape of an array is indicated by a tuple of integers
# i.e. height, width
# zeroes_like creates an array of zeroes with the same shape as the image
# i.e. same number of rows and columns
# since all of the pixels are zeroes ==> completely black mask
# fillPoly fills an area bounded by several polygons to be the intensity
# that we specify
# in our case we have only one polygon ==> the triangle shape
# OpenCV read color images as Blue, Green, Red (BGR)
# To set the color as white:
# cv2.fillPoly(mask, polygons, [255, 255, 255])

# 5. Apply a bitwise and between the mask and the canny (derivative) image
# Each pixel on the image has an intensity
# which is effectively a bunch of 1s and 0s in binary
# E.g of a bitwise and ==> 0 1 1 0 0 1 & 1 1 0 0 1 0 = 0 1 0 0 0 0
# The mask we created is completely filled with 1s
# With the bitwise & pixel by pixel ==> we chop off other edges
# and retain just the path we are interested in


# 6. Detecting Straight lines in an image ==> Hough transform
# eqn of a straight line ==> y = mx + b
# In hough space, we plot b against m
# A single x,y coordinate point
# can be passed through by many lines with varying b,m
# So hough space represents possible combinations of b.m as a line
# for each x,y coordinate point in the cartesian space
# Interestion point between 2 lines in hough space represents
# b,m combination line that passes through the two
# points x1,y1 and x2, y2
# But again there can be multiple lines with other b,m combinations
# passing through x1, y1 and x2, y2
# To choose the best fit line to pass through the points
# we make grids in the hough space and
# the grid with the most number of intersection point is
# the one from which we choose a possible b,m combination
# for the best fit line
# A vote is cast for each point of intersection
# Vertical lines can pass through a x,y coordinate point
# But that has m = infinity
# This, we cant represent in hough space
# Thus instead of y = mx + b for lines
# We adopt expression of a line using polar coordinates
# where p, called as rho, is the perpendicular distance
# from the origin to the line or point on the line and
# teta is the angle of inclination from the positive x axis to the rho line
# p = xcos(teta) + gsin(teta)
# where x, g is the coordinates of a point
# So in the hough space, we now plot the possible (p, teta)
# for each point in the cartesian space
# To draw a line you just need two points

# 7. Optimising
# Now we already have the straight lines displaying on the colored image
# But some lines are segmnented and we want a continuous line on the
# left and rigth of image
# As x increases ==> y increases ==> positive slope ==> m > 0
# image.shape() ==> prints a tuple containing
# the height, width and number of color channels in the image
# x = (y - b) / m


# IMAGE PROCESSING


# loads the image
# returns a multi - dimensional np array with the
# relative intensities of each pixel of the image
# specify filename
# image = cv2.imread("test_image.jpg")

# copy the array into a new variable
# lane_image = np.copy(image)

# image wih pixel axis
# canny_image = cannyAlgo(lane_image)

# displays only region of interest
# cropped_image = regionOfInterest(canny_image)


# detected straight lines on cropeed image
# 1st argument ==> image in which you want to detect line
# 2nd (p) and 3rd (teta) agruments specify the resolution of the
# Hough accumulator array or the grid which is a 2d array
# that we use to collect votes for the best fit line
# Each bin or grid has a distinct p, in pixels and teta value in radians
# 4th argument ==> we specify threshold on which bin to choose
# i.e. min num of votes to detect a line
# we choose the bin with >=100 votes
# 5th argument is a placeholder array for the output
# Any detected line less than 40 pixels is rejected
# Between segmented lines detected on image analysed, if the gap is <= 5,
# we join them together
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
#                         np.array([]), minLineLength=40, maxLineGap=5)

# array containing two lines
# averaged_Lines = averageSlopeIntercept(lane_image, lines)

# line_image = displayLines(lane_image, averaged_Lines)

# addWeighted takes the sum of the color lane_image
# and the line_image which has the lines
# Multiply all pixel intensities in lane_image by 0.8
# this makes it darker
# multiply line_image by 1
# add 1 to the sum of the pixel intensities of the two images
# comboImage = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# show the image
# cv2.imshow("result", comboImage)

# displays the image for a specified amt of milli seconds
# setting it to 0 ==> makes the image stay displayed
# # infinelty until we hit a key
# cv2.waitKey(0)


# -->ord('q') returns the Unicode code point of q

# -->cv2.waitkey(1) returns a 32-bit integer corresponding to the pressed key

# -->& 0xFF is a bit mask which sets the left 24 bits to zero,
# because ord() returns a value betwen 0 and 255,
# since your keyboard only has a limited character set

# -->Therefore, once the mask is applied, it is
# then possible to check if it is the corresponding key.

# Generally in the OpenCV tutorials and blogs,
# it is a general convention to use "q" key for halting
# any indefinite operation such as capturing frames from camera in your case.
# In your case, the program indefinitely checks at each iteration
# if "q" key is being pressed using
# cv2.waitKey(1) & 0xFF == ord('q') statement.
# If True then it simply brakes the infinite while loop.
#  You can set it to any key of your choice.
