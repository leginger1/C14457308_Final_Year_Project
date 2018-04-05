import cv2
import logging as log
import math
import thresholding as thresh
import sympy as sym

# Person class to hold all details about person
class Person:

    # Bounding rectangles
    boundingRectangles = []
    predictedBoundingRectangle = None

    # Convex hulls representing the person
    convexHulls = []

    # Splash values of the area of the person
    splashValue = 0

    # Posture value of the person
    postureValue = None

    # Bobbing measurement of the person
    bobbingValue = None

    # Speed measurement
    speed = 0

    def __init__(self):
        self.boundingRectangles = []
        self.predictedBoundingRectangle = None
        self.convexHulls = []
        self.splashValue = 0
        self.postureValue = None
        self.bobbingValue = None
        self.speed = 0

    # Function to update the person object
    def updatePerson(self, convexHull):
        self.updateConvexHull(convexHull)
        self.updateBoundingRect(convexHull)
        self.calcSpeed()


    #Funtion to draw person details onto the frame
    def drawDetails(self, frame, index):

        # Defining colours
        red = (0,0,255)
        green = (0,255,0)
        blue = (255,0,0)

        # Setting the font for the text
        font = cv2.FONT_HERSHEY_PLAIN

        # Making the text string
        text = "Person Index: " + str(index) + "   Speed " + str(self.speed) + "   Splash Value: " + str(self.getSplashValue())

        # Getting the location of the person
        location = self.getRectCenter(self.getBoundingRect())

        # Getting ellipse for the person
        ellipse = thresh.getEllipse(self.getConvexHull())

        current_colour = red

        if self.speed < 5:
            current_colour = blue

        if self.splashValue > 100:
            current_colour = green

        # Drawing ellipse
        thresh.drawEllipse(frame, ellipse, current_colour)

        # Putting the text onto the frame
        cv2.putText(frame, text, location, font, 0.5, green, 1, cv2.LINE_AA)

        return frame

    # Function to check if an ellipse is the same person or not
    def samePerson(self, convexHull, numReferenceFrames):
        rectangle = cv2.boundingRect(convexHull)

        # if len(self.boundingRectangles) >= numReferenceFrames:
        #     predictedRectangle = self.predictNextRect(numReferenceFrames)
        #     percent_overlap = self.calcPercentageOverlap(predictedRectangle, rectangle)
        #     return percent_overlap
        #
        # else:

        previous_rectangle = self.getBoundingRect()
        percent = self.calcAreaOverlapSamePlacePercent(previous_rectangle, rectangle)

        return percent


    # Function to predict the next bounding rectangle for a person
    def predictNextRect(self, numReferenceFrames):

        x_coord, y_coord = self.predictNextLocation(numReferenceFrames)
        width = self.calcAvgWidth(numReferenceFrames)
        height = self.calcAvgHeight(numReferenceFrames)

        rectangle = (x_coord, y_coord, width, height)

        return rectangle


    # Function to predict if where the next ellipse for that person should be
    def predictNextLocation(self, numReferenceFrames):
        # Getting the average speed of the swimmer
        (speedX, speedY) = self.calcAvgSpeed(numReferenceFrames)

        # Getting the last location of the swimmer
        (currentX, currentY) = self.getRectCenter(self.getBoundingRect())

        # Calculating the next location of the swimmer based on their current speed
        nextX = currentX + speedX
        nextY = currentY + speedY

        return (nextX, nextY)


    # Function to calculate the average speed in the X and Y axis of the person over a number of reference frames
    def calcAvgSpeed(self, numReferenceFrames):

        sumXSpeed = 0
        sumYSpeed = 0

        for index in range(1, numReferenceFrames):
            rectangle1 = self.getBoundingRect(index - 1)
            rectangle2 = self.getBoundingRect(index)
            sumXSpeed += rectangle1[0] - rectangle2[0]
            sumYSpeed += rectangle1[1] - rectangle2[1]

        avgXSpeed = sumXSpeed / numReferenceFrames
        avgYSpeed = sumYSpeed / numReferenceFrames

        avgSpeed = math.sqrt(pow(avgXSpeed, 2) + pow(avgYSpeed, 2))
        self.avgSpeed = avgSpeed

        return (avgXSpeed, avgYSpeed)


    # Function to calculate the speed of a person
    def calcSpeed(self):
        if len(self.boundingRectangles) >= 2:
            speed = self.dist(self.getBoundingRect(), self.getBoundingRect(len(self.boundingRectangles)-2))
            self.speed = speed
        else:
            self.speed = 0


    # Function to calculate the average width of bounding rectangle over a number of reference frames
    def calcAvgWidth(self, numReferenceFrames):

        sumWidth = 0

        for index in range(0, numReferenceFrames):
            rectangle = self.getBoundingRect(index)
            sumWidth += rectangle[2]

        avgWidth = sumWidth / numReferenceFrames

        return avgWidth


    # Function to calculate the average height of bounding rectangle over a number of reference frames
    def calcAvgHeight(self, numReferenceFrames):

        sumHeight = 0

        for index in range(0, numReferenceFrames):
            rectangle = self.getBoundingRect(index)
            sumHeight += rectangle[3]

        avgHeight = sumHeight / numReferenceFrames

        return avgHeight


    # Function to get the distance between two points
    def dist(self, rectangle1, rectangle2):

        X1 = rectangle1[0]
        Y1 = rectangle1[1]

        X2 = rectangle2[0]
        Y2 = rectangle2[1]

        # Distance formula sqrt((X2-X1)^2 + (Y2-Y1)^2)
        distance = abs(math.sqrt(math.pow((X2-X1),2) + math.pow((Y2-Y1),2)))

        return distance


    # Function to get an image of the person
    def getPersonArea(self, frame):
        (x,y,w,h) = self.getBoundingRect()
        return frame[y:y+h,x:x+h,:]


    # Function to calculate the splash value
    def calcSplashValue(self, frame, splashThreshold):
        # The area that
        splashArea = self.getPersonArea(frame)

        splashArea = thresh.convertColourSpace(splashArea, cv2.COLOR_BGR2GRAY)

        # Number of pixels seen as "splash pixels"
        splashPixelNum = 0

        # Looping through pixels in swimmer bounding box
        for row in splashArea:

            for pixel in row:

                # Splash pixel indicator
                splashPixel = True

                if pixel < splashThreshold:
                    splashPixel = False

                if splashPixel:
                    splashPixelNum += 1

        # Deciding if the value needs to be set or updated (has it been set before?)
        self.updateSplashValue(splashPixelNum)


    # Function to get the convex hull
    def getConvexHull(self, index = len(convexHulls)-1):
        return self.convexHulls[index]


    # Function to update current convex hull and add old one to list of previous
    def updateConvexHull(self, convexHull):
        self.convexHulls.append(convexHull)


    # Function to get the splash value
    def getSplashValue(self):
        return self.splashValue


    # Function to update the splash value
    def updateSplashValue(self, splashValue):
        self.splashValue = splashValue


    # Function to get bounding rectangle for the person (x,y,w,h)
    def getBoundingRect(self, index = len(boundingRectangles)-1):
        return self.boundingRectangles[index]


    # Function to update the bounding rectangles
    def updateBoundingRect(self, convexHull):
        boundingRect = cv2.boundingRect(convexHull)
        self.boundingRectangles.append(boundingRect)


    # Function to get the center of a given rectangel
    def getRectCenter(self, rectangle):
        return (rectangle[0], rectangle[1])


    # Function to get the percentage of overlap of one rectangle with respect to another
    def calcPercentageOverlap(self, rectangle1, rectangle2):
        areaOverlap = self.calcAreaOverlap(rectangle1, rectangle2)
        areaRect = self.calcAreaRect(rectangle1)

        percent_overlap = (areaOverlap * 100) / areaRect

        return percent_overlap


    # Function to calculate the are of overlap of two rectangles in the same place
    def calcAreaOverlapSamePlacePercent(self, rectangle1, rectangle2):
        # Setting both rectangles to the same location
        rectangle1_area = self.calcAreaRect(rectangle1)
        rectangle2_area = self.calcAreaRect(rectangle2)

        if rectangle1_area > rectangle2_area:
            percent = (rectangle2_area * 100) / rectangle1_area
        else:
            percent = (rectangle1_area * 100) / rectangle2_area

        return percent


    # Function to get the area of intersection between two rectangles
    def calcAreaOverlap(self, rectangle1, rectangle2):
        # Rectangle 1 parameters
        rectangle1_left = rectangle1[0]
        rectangle1_right = rectangle1_left + rectangle1[2]
        rectangle1_top = rectangle1[1]
        rectangle1_bottom = rectangle1_top + rectangle1[3]

        # Rectangle 2 parameters
        rectangle2_left = rectangle2[0]
        rectangle2_right = rectangle2_left + rectangle2[2]
        rectangle2_top = rectangle2[1]
        rectangle2_bottom = rectangle2_top + rectangle2[3]

        # Calculating the x and y overlaps
        x_overlap = max(0, min(rectangle1_right, rectangle2_right) - max(rectangle1_left, rectangle2_left))
        y_overlap = max(0, min(rectangle1_bottom, rectangle2_bottom) - max(rectangle1_top, rectangle2_top))

        # Calculating the area of overlap
        overlap_area = x_overlap * y_overlap

        return overlap_area


    # Function to calculate the area of a rectangle
    def calcAreaRect(self, rectangle):

        rect_width = rectangle[2]
        rect_height = rectangle[3]

        rect_area = rect_width * rect_height

        return rect_area