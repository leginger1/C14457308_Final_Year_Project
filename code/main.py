# Defining imports
import cv2
import video as vid
import thresholding as thresh
import person as p

print(cv2.__version__)

# Defining video paths
lengthsVideoPath = "../media/video/edited/lengths.avi"
lengthsVideoPath2 = "../media/video/edited/lengths2.avi"
splashingVideoPath = "../media/video/edited/splashing.avi"
comboVideoPath = "../media/video/edited/combo.avi"

# Opening video capture
capture = vid.openVideoCapture(lengthsVideoPath2)

# Declaring background subtractors
subtractorMOGDefault = thresh.createMOGSubtractor()
subtractorMOG2 = thresh.createMOG2Subtractor(1000, 25, False)
subtractorMOG2Default = thresh.createMOG2Subtractor()
subtractorGMG = thresh.createGMGSubtractor(150, 0.75)
subtractorGMGDefault = thresh.createGMGSubtractor()
subtractorGSOC = thresh.createGSOCSubtractor()

# Declaring Kernels
smallKernel = thresh.createStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mediumKernel = thresh.createStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
largeKernel = thresh.createStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# Tuple to hold kernels
kernels = (smallKernel, mediumKernel, largeKernel)

# List to hold all the contours, convexHulls, images, and people objects
contours = []
convexHulls = []
images = []
people = []

# Frame count of the video
frameCount = 0

# Boolean to decide if the program should be running or not
running = True


# Function to identify people
def identifyPeople(convexHulls, people, personLimit):

    # List to hold percentage likely hood for convex hull with respect to each person
    people_percents = []
    people_list = people[:]

    # If no people currently exist in the list, (i.e. the number of people in the list is zero) each convex hull becomes a person object
    if len(people_list) == 0:

        # Looping through each convex hull
        for hull in convexHulls:

            # Breaking from loop and returning people list if person limit is reached
            if len(people_list) >= personLimit:
                break

            # Creating a person object from each convex hull
            person = p.Person()
            person.updatePerson(hull)

            # Appending the person object to the list of people
            people_list.append(person)

        # Returning the list of people
        return people_list

    # Else if there are people in the list of people
    else:
        # If there are convex hulls in the list,
        if len(convexHulls) > 0:

            # Number of reference frames to use
            referenceFrames = 3

            # Getting the percent likely hoods
            # Looping through each of the people in the list
            for person in people:

                # Empty list to hold the percent likely hood of each convex hull belonging to the person
                percents = []

                # Looping through each convex hull
                for hull in convexHulls:

                    # Getting the percent likely hood for the current convex hull to belong to the current person
                    percent = person.samePerson(hull, referenceFrames)

                    # Getting the distance between the last location of the person and the current convex hull
                    distance = person.dist(person.getBoundingRect(), cv2.boundingRect(hull))

                    # If the distance is greater than 100 cant possibly be the person
                    if distance > 100:
                        percent = 0

                    # Appending the percent likely hood to the list of percents for the person
                    percents.append(percent)

                # Appending the percent likely hoods for the given person to the list
                people_percents.append(percents)

            # Adding convex hulls to the people objects based on the percent likely hoods
            for percents in people_percents:

                # Getting the index of the percents list in people_percents
                index = people_percents.index(percents)

                # Getting the largest percentage for the current index
                largest = max(percents)

                # If the largest percentage is greater than 50 percent
                if largest > 60:

                    # Getting the convex hull at the same index as the largest value
                    hull = convexHulls[percents.index(largest)]

                    # Updating the person object with the convex hull
                    people[index].updatePerson(hull)

        # Return the list of people
        return people


# Loop till end of video
while(running):

    frameCount += 1

    frame = vid.readFrame(capture)

    # Reopening video if it reaches the end
    if frameCount%capture.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        capture = vid.openVideoCapture(lengthsVideoPath2)

    images.clear()

    if frameCount % 3 == 0:
        # Applying background subtractor
        subFrame = thresh.applySubtractor(frame, subtractorGSOC)

        if frameCount >= 1000:
            # Getting contours and convex hulls
            contours = thresh.findContours(subFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            convexHulls = thresh.getConvexHulls(contours, convexHulls, 10000)

            # Sorting the convex hulls by size
            convexHulls = sorted(convexHulls, key = cv2.contourArea, reverse = True)

            # Identifying people in the scene
            people = identifyPeople(convexHulls, people, personLimit=3)

            for person in people:
                # Calculating the splash value for person object
                person.calcSplashValue(frame, 200)
                # Drawing all of the details for the person object (Index, speed, splash value)
                person.drawDetails(frame, people.index(person))

        # Display images
        images = thresh.addImageToList(thresh.convertColourSpace(subFrame, cv2.COLOR_GRAY2BGR), images)
        images = thresh.addImageToList(frame, images)

        # Concatinating the binary image and the normal image into one
        frames = thresh.concatImages(images, 1)

        # Displaying the video frame
        vid.showFrame(frames, "temp", 45)

    print(frameCount)


# Pausing the program to stop sudden close
cv2.waitKey(0)

# Closing the video capture and windows
vid.closeVideoCapture()
vid.closeAllWindows()


