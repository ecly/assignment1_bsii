# import the necessary packages
import argparse
import imutils
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance

from RegionProps import RegionProps

def DetectPupil(cropped_frame):
    #If the frame is non-existing.
    if len(cropped_frame[0]) < 1:
        return 0,0
        
    #Input values from original function
    threshold = 79
    pupilMinimum = 10
    pupilMaximum = 50
    
    """
    Given an image, return the coordinates of the pupil candidates.
    """
    # Create the output variable.
    bestPupil = -1 

    # Grayscale image.
    grayscale = cropped_frame.copy()
    if len(grayscale.shape) == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    # Define the minimum and maximum size of the detected blob.
    pupilMinimum = int(round(math.pi * math.pow(pupilMinimum, 2)))
    pupilMaximum = int(round(math.pi * math.pow(pupilMaximum, 2)))

    # Create a binary image.
    _, thres = cv2.threshold(grayscale, threshold, 255,
                                 cv2.THRESH_BINARY_INV)

    kernel = np.ones((6,6), np.uint8)
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

    # Find blobs in the input image.
    _, contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    props = RegionProps()
    largestArea = 0
    areas = list()
    centers = list()
    for contour in contours:
        prop = props.calcContourProperties(contour, ["Centroid", "Area", "Extend"])
        ellipse = cv2.fitEllipse(contour) if (len(contour) > 4) else cv2.minAreaRect(contour)

        centroid = prop.get("Centroid")
        area = prop.get("Area")
        extend = prop.get("Extend")
        if(area > pupilMinimum and area < pupilMaximum and extend > 0.5):
            centers.append(centroid)
            areas.append(area)

            if (area > largestArea):#define best pupil as largest area
                largestArea = area
                bestPupil = len(centers) - 1

    #if we didn't find a pupil candidate
    if bestPupil == -1:
        return 0,0
    
    pupilCenter = centers[bestPupil]
    PupilX = pupilCenter[0]
    PupilY = pupilCenter[1]
    
    return PupilX, PupilY