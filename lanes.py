from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def cannyAlgo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# loads the image
# returns a multi - dimensional np array with the
# relative intensities of each pixel of the image
# specify filename
# intensities = color
# ranges from 0 ( black ) to 255( white )
# image is an array
image = cv2.imread("test_image.jpg")

# copy the array into a new variable
lane_image = np.copy(image)

canny = cannyAlgo(lane_image)


# show the image
plt.imshow(canny)

# displays the image for a specified amt of milli seconds
# setting it to 0 ==> makes the image stay displayed
# # infinelty until we hit a key
plt.show()


# conda install -c conda-forge opencv
# check pip installation online

# EDGE DETECTION = sharp change of intensity b/w adjacent pixels
# sharp change = steep gradient
# width or x axis ==> number of cols in an image
# height or y axis ==> number of rows in an image
# The image can be viewed as a function of x and y ==> f(x,y)
# The derivative ==> gradient


# 1. convert image to grayscale

# 2. reduce noise using GaussianBlur
# Each pixel value is set to the
# weighted average of the surrouding pixels
# using a 5x5 kernel with zero deviation

# 3. Find derivative
# low threshold pixels rejected
# high threshold pixels accepted as edge pixel
# those between low and high will be accepted only if
# that pixel is connected to a edge pixel
# use a low - high threshold of 1 : 3
# Gradients that exceed the high threshold
# as traced as bright pixels in white
# Gradients that are small and below the low threshold are black
