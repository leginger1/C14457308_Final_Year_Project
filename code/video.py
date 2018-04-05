import cv2
import logging as log
import sys


# Function to open video capture
def openVideoCapture(fileName):
    log.info("Opening video capture.")
    return cv2.VideoCapture(fileName)


# Function for closing video capture
def closeVideoCapture(capture):
    log.info("Releasing video capture.")
    capture.release()


# Function for reading video frame
def readFrame(capture):
    log.info("Reading video frame.")
    read, frame = capture.read()

    if not read:
        log.info("Failed to read frame. End of video reached.")
        closeVideoCapture(capture)
        sys.exit()

    return frame


# Function for displaying video frame
def showFrame(frame, windowName = "Default Window", fps = 30):
    log.info("Showing video frame")
    cv2.waitKey(fpsMillisWait(fps))
    cv2.imshow(windowName, frame)


# Function for calculating delay between frames
def fpsMillisWait(fps):
    return round(1000 / fps)

# Function to close all windows
def closeAllWindows():
    log.info("Closing all windows")
    cv2.destroyAllWindows()