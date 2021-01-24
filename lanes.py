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
    heightOfImage = image.shape[0]
    polygons = np.array([
        [(200, heightOfImage), (1100, heightOfImage), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, [255, 255, 255])
    return mask


# loads the image
# returns a multi - dimensional np array with the
# relative intensities of each pixel of the image
# specify filename
image = cv2.imread("test_image.jpg")

# copy the array into a new variable
lane_image = np.copy(image)

canny = cannyAlgo(lane_image)


# show the image
cv2.imshow("result", regionOfInterest(canny))

# displays the image for a specified amt of milli seconds
# setting it to 0 ==> makes the image stay displayed
# # infinelty until we hit a key
cv2.waitKey(0)


# THE LOGIC BEHIND THE CODE

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

# 1. convert image to grayscale

# 2. reduce noise using GaussianBlur
# Each pixel value is set to the
# weighted average of the surrouding pixels
# using a 5x5 kernel with zero deviation

# 3. Find derivative ==> Canny
# low threshold pixels rejected
# high threshold pixels accepted as edge pixel
# those between low and high will be accepted only if
# that pixel is connected to a edge pixel
# use a low - high threshold of 1 : 3
# Gradients that exceed the high threshold
# as traced as bright pixels in white
# Gradients that are small and below the low threshold are black


# 4. Set the region of interest
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
