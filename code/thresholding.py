import cv2
import cv2.bgsegm as bg
import logging as log
import math
import numpy as np

# Function to create mog2 background subtractor
def createMOG2Subtractor(history = 500, varThreshold = 16, detectShadows = True):
    log.info("Creating MOG2 subtractor")
    return cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)


# Function to create mog background subtractor      history, nmixtures, backgroundRatio, noiseSigma
def createMOGSubtractor():
    log.info("Create MOG subtractor")
    return bg.createBackgroundSubtractorMOG()


# Function to create cnt background subtractor
def createCNTSubtractor(minPixelStability = 15, useHistory = True, maxPixelStability = 15*60, isParallel = True):
    log.info("Creating CNT subtractor")
    return bg.createBackgroundSubtractorCNT(minPixelStability, useHistory, maxPixelStability, isParallel)


# Function to create GSOC background subtractor   mc, nSamples, replaceRate, propagationRate, hitsThreshold, alpha, beta, blinkingSupressionDecay, blinkingSupressionMultiplier, noiseRemovalThresholdFacBG, noiseRemovalThresholdFacFG
def createGSOCSubtractor():
    log.info("Creating GSOC subtractor")
    return bg.createBackgroundSubtractorGSOC()


# Function to create LSBP background subtractor
def createLSBMSubtractor():
    log.info("Creating LSBP subtractor")
    return bg.createBackgroundSubtractorLSBP()


# Function to create GMG background subtractor
def createGMGSubtractor(initializationFrames = 150, threshold = 0):
    log.info("Creating GMG subtractor")
    return bg.createBackgroundSubtractorGMG(initializationFrames, threshold)


# Function to apply subtractor to frame
def applySubtractor(frame, subtracter):
    log.info("Subtracting background from frame")
    return subtracter.apply(frame)


# Function to apply averaging filter to video frame
def applyAveragingFilter(frame, size):
    log.info("Applying an averaging filter to the frame")
    return cv2.blur(frame, size)


# Function to apply gaussian blur to video frame
def applyGaussianBlur(frame, size):
    log.info("Applying a gaussian blur to the frame")
    return cv2.GuassianBlur(frame, size)


# Function to apply median blur
def applyMedianBlur(frame, size):
    log.info("Applying a median blur to the frame")
    return cv2.medianBlur(frame, size)


# Function to apply bilateral filter
def applyBilateralFilter(frame, d, sigmaColor, sigmaSpace):
    log.info("Applying a bilaterial filter to the frame")
    return cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)


# Function to create a structuring element ELLIPSE or RECTANGLE shape
def createStructuringElement(shape, kernelSize):
    log.info("Creating a structuring element (kernel)")
    return cv2.getStructuringElement(shape, kernelSize)


# Function to dialate frame of video
def dialateFrame(frame, kernel):
    log.info("Dilating the frame")
    return cv2.dilate(frame, kernel)


# Function to erode frame of video
def erodeFrame(frame, kernel):
    log.info("Eroding the frame")
    return cv2.erode(frame, kernel)


# Function to open a frame of video (erode then dialate)
def openFrame(frame, kernel):
    log.info("Opening the frame")
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)


# Function to close a frame of a video (dialate then erode)
def closeFrame(frame, kernel):
    log.info("Closing the frame")
    return  cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)


# Fucntion to convert the colour space of a frame of a video
def convertColourSpace(frame, colourSpace):
    log.info("Converting the colour space of the frame")
    return cv2.cvtColor(frame, colourSpace)


# Function to find contours in a binary image
def findContours(frame, mode, method):
    log.info("Finding the contours of a binary image")
    image, contours, hierichary = cv2.findContours(frame, mode, method)
    return contours


# Function to get list of convex hulls
def getConvexHulls(contours, convexHulls, areaThreshold = 0):
    log.info("Getting convex hulls for a list of contours")
    convexHulls.clear()
    for contour in contours:
        hull = getConvexHull(contour)
        if len(hull) >= 5:
            ellipse = getEllipse(hull, areaThreshold)
            if ellipse is not None:
                convexHulls.append(getConvexHull(contour))

    # print("Convex Hulls")
    # print(convexHulls)
    return convexHulls


# Function to get the convex hull of a contour
def getConvexHull(contour):
    log.info("Finding the convex hull of a contour")
    return cv2.convexHull(contour)


# Function to get ellipses for an list of convex hulls
def getEllipses(convexHulls, ellipses, areaThreshold = 0):
    log.info("Getting list of ellipses for convex hulls")
    ellipses.clear()
    for hull in convexHulls:
        # Hull must have more than five points to fit ellipse
        if len(hull) >= 5:
            ellipse = getEllipse(hull, areaThreshold)
            if ellipse is not None:
                ellipses.append(ellipse)
    return ellipses


# Function to get an ellipse around a convex hull
def getEllipse(convexHull, areaThreshold = 0):
    log.info("Getting an ellipse around a contour")
    ellipse = cv2.fitEllipse(convexHull)
    rect = cv2.boundingRect(convexHull)

    (MA,ma) = ellipse[1]

    if areaEllipse(MA, ma) > areaThreshold:
        log.info("Returning ellipse of appropiate size")
        return ellipse

    return None


# Function to get the area of an ellipse
def areaEllipse(majorAxis, minorAxis):
    log.info("Getting the area of an ellipse")
    return math.pi * majorAxis * minorAxis


# Function to draw an array of ellipses on a frame
def drawEllipses(frame, ellipses):
    log.info("Drawing list of ellipses")
    for ellipse in ellipses:
        frame = drawEllipse(frame, ellipse)
    return frame


# Function to draw an ellipse on a frame
def drawEllipse(frame, ellipse, colour = (255, 0, 0)):
    log.info("Drawing ellipse")
    return cv2.ellipse(frame, ellipse, colour, 2)


# Function to add an image to a list
def addImageToList(image, imageList):
    imageList.append(image)
    return imageList


# Function to clear images from list
def clearImageList(imageList):
    imageList.clear()
    return imageList


# Funtion to put two images together (axis: 0=vertical, 1=horizontal)
def concatImages(imagelist, axis):
    imagetuple = tuple(imagelist)
    return np.concatenate(imagetuple, axis)