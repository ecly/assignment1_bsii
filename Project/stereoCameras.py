#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : stereoCameras.py                                         -->
#<!-- Description: Script to work with stereo cameras setups                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 25/04/2017                                               -->
#<!-- Change     : 25/04/2017 - Creation of this script                     -->
#<!-- Review     : 25/04/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017042501 $"

########################################################################
import cv2
import numpy as np

from collections import deque
from RecordVideo import RecordVideo
import time

from CaptureVideo.CaptureVideo import CaptureVideo

########################################################################
def findChessboardCorners(image):
    """Finds the positions of internal corners of the chessboard."""
    # Processed image
    gray = image.copy()

    # Finds the positions of internal corners of the chessboard.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    retval, corners = cv2.findChessboardCorners(gray, (9, 6), flags=flags)

    if retval:
        # Check if the input image is a grayscale image.
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # Refines the corner locations.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the detected chessboard.
        cv2.drawChessboardCorners(image, (9, 6), corners, True)
    else:
        corners = None

    # Garbage Collector.
    del gray

    # Return the final result.
    return corners

def calculatePattern():
    """Creates a standard vectors of the calibration pattern points."""
    # Create the main vector.
    square_size  = 1.0
    objectPoints = np.zeros((np.prod((9, 6)), 3), np.float32)
    objectPoints[:, :2] = np.indices((9, 6)).T.reshape(-1, 2)
    objectPoints *= square_size

    # Return the final result.
    return objectPoints

def calibrate(leftCorners, rightCorners, objectPoints):
    # Global variables.
    global map1
    global map2

    #<!--------------------------------------------------------------------------->
    #<!--                            YOUR CODE HERE                             -->
    #<!--------------------------------------------------------------------------->
    _, mtx1, dist1, _, _ = cv2.calibrateCamera(objectPoints, leftCorners, imageSize, None, None)
    _, mtx2, dist2, _, _ = cv2.calibrateCamera(objectPoints, rightCorners, imageSize, None, None)
    
    _, mtx1, dist1, mtx2, dist2, R, T, E, F =cv2.stereoCalibrate(objectPoints, leftCorners, rightCorners, mtx1, dist1, mtx2, dist2, imageSize)
    skew = crossProductMatrix(T)
    essential = skew * R
    
    r1, r2, p1, p2, q, validPix1, validPix2,  = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, imageSize, R, T)
    map1 = cv2.initUndistortRectifyMap(mtx1, dist1, r1, p1, imageSize, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(mtx2, dist2, r2, p2, imageSize, cv2.CV_32FC1)
    #<!--------------------------------------------------------------------------->
    #<!--                                                                       -->
    #<!--------------------------------------------------------------------------->

def crossProductMatrix(t):
    """Estimating the skew symmetric matrix."""
    #<!--------------------------------------------------------------------------->
    #<!--                            YOUR CODE HERE                             -->
    #<!--------------------------------------------------------------------------->
    #Calculate skew matrix as seen in [SO]
    t1 = t[0][0]
    t2 = t[1][0]
    t3 = t[2][0]
    skew = np.array([[0 , -t3,   t2],
                    [t3 ,   0,  -t1],
                    [-t2,  t1,    0]])
    
    return skew
    #<!--------------------------------------------------------------------------->
    #<!--                                                                       -->
    #<!--------------------------------------------------------------------------->

########################################################################
# Create a capture video object.
# Tips: You can add a new video capture device or video file with the method
# CaptureVideo.addInputVideo().
capture = CaptureVideo(isDebugging=True)
capture.addInputVideo("video01.m4v", size=(640, 480), framerate=5.)
capture.addInputVideo("video02.m4v", size=(640, 480), framerate=5.)

# Creates a window to show the stereo images.
cv2.namedWindow("Stereo", cv2.WINDOW_AUTOSIZE)
print "WARNING: When the chessboard will be detected, press \"S\" in your keyboard.\n"

# Global variables.
leftCorners  = []
rightCorners = []
objectPoints = []
imageSize    = (0, 0)
map1 = []
map2 = []

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    retval, frames = capture.getFrames()

    # Check if there is a valid frame.
    if not retval:
        break

    # Get the image size.
    
    imageSize = frames[0].shape[::-1][1:3]
    
    # Find the pattern in both images.
    left  = findChessboardCorners(frames[0])
    right = findChessboardCorners(frames[1])

    # Create the stereo image.
    stereo = np.hstack((frames[0], frames[1]))

    # Undistorted image.
    if (len(map1) == 2):
        left = cv2.remap(frames[0], map1[0], map1[1], cv2.INTER_LINEAR)
        right = cv2.remap(frames[1], map2[0], map2[1], cv2.INTER_LINEAR)
        undistorted = np.hstack((left, right))
        stereo = np.vstack((stereo, undistorted))
        stereo = cv2.resize(stereo, (0, 0), fx=0.5, fy=0.5)

    # Check what the user wants to do.
    key = cv2.waitKey(1)

    # Esc or letter "q" key.
    if key == 27 or key == ord("q"):
        break

    # Letter "s" key.
    elif key == ord("s"):
        if left is not None and right is not None:
            leftCorners.append(left)
            rightCorners.append(right)
            objectPoints.append(calculatePattern())

            size = len(leftCorners)
            print "Detected Chessboard: %02d of 15." % (size)
            if size == 5:
                calibrate(leftCorners, rightCorners, objectPoints)
                leftCorners  = []
                rightCorners = []
                objectPoints = []

    # Letter "w" key.
    elif key == ord("w"):
        if (len(map1) == 2):
            cv2.imwrite("left.jpg", left)
            cv2.imwrite("right.jpg", right)

    # Letter "c" key.
    elif key == ord("c"):
        leftCorners  = []
        rightCorners = []
        objectPoints = []
        map1 = []
        map2 = []


    
    
    # Display the resulting frame.
    cv2.imshow("Stereo", stereo)
    

# When everything done, release the capture object.
del capture
cv2.destroyAllWindows()
