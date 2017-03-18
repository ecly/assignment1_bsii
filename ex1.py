#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : ex1.py                                                   -->
#<!-- Description: Script to locate the pupil in the input videos           -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 13/02/2017                                               -->
#<!-- Change     : 13/02/2017 - Creation of this script                     -->
#<!-- Review     : 13/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021301 $"

########################################################################
import argparse
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

from EyeFeatureDetector        import EyeFeatureDetector
from CaptureVideo.CaptureVideo import CaptureVideo
from RecordVideo.RecordVideo   import RecordVideo

########################################################################
# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, help="Input Video Source")
ap.add_argument("-o", "--output", required=True, help="Output Video Filename")
args = vars(ap.parse_args())

# Create the Matplotlib window.
# There is a bug on macOS that it is not possible to open an OpenCV windows
# before openning a Matplotlib windows.
fig = plt.figure()
plt.close()

def onSlidersChange(self, dummy=None):
    """ Handle updates when slides have changes."""
    global slidersVals
    slidersVals["pupilThr"] = cv2.getTrackbarPos("pupilThr", "trackbars")
    slidersVals["pupilMinimum"]  = cv2.getTrackbarPos("pupilMinimum", "trackbars")
    slidersVals["pupilMaximum"]  = cv2.getTrackbarPos("pupilMaximum", "trackbars")

# Define the input video source.
uid = 0
if args["input"] != None:
    if type(args["input"]) is str and args["input"].isdigit():
        uid = int(args["input"])
    else:
        uid = "Inputs/" + args["input"]

# Define the detector arguments.
slidersVals = {}
slidersVals["pupilThr"] = 0
slidersVals["pupilMinimum"]  = 10
slidersVals["pupilMaximum"]  = 50

# Create the eye feature detector.
detector = EyeFeatureDetector()

# Create an OpenCV window and some trackbars.
cv2.namedWindow("trackbars", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("pupilThr", "trackbars", 0, 255, onSlidersChange)
cv2.createTrackbar("pupilMinimum", "trackbars", 10, 40,  onSlidersChange)
cv2.createTrackbar("pupilMaximum", "trackbars", 50, 100, onSlidersChange)
cv2.imshow("trackbars", np.zeros((1, 640), np.uint8))

# Create a capture video object.
# Tips: You can add a new video capture device or video file with the method
# CaptureVideo.addInputVideo().
capture = CaptureVideo(isDebugging=True)
capture.addInputVideo(uid, size=(640, 480), framerate=30.)

# Create a record video object.
# Tips: You can add a new output video file with the method
# RecordVideo.addOutputVideo().
record = RecordVideo(True)
record.addOutputVideo("Outputs/" + args["output"], size=(640, 480),
                      framerate=30., isColor=True)

# Start the record thread.
record.startThread()

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    retval, frames = capture.getFrames()

    # Check if there is a valid frame.
    if not retval:
        break

    # Get the coordinates of the pupil candidates.
    pupils = detector.getPupil(frames, slidersVals["pupilThr"],
                               slidersVals["pupilMinimum"],
                               slidersVals["pupilMaximum"])

    # Get the processed image.
    processed = detector.getProcessedImage(frames, slidersVals["pupilThr"],
                                           pupilsEllipses=pupils[0],
                                           pupilsCenters=pupils[1],
                                           bestPupil=pupils[2])
    record.writeFrames(processed)

    # Display the resulting frame.
    cv2.imshow("ex1", processed)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

# Start the record thread.
record.stopThread()
while record.IsRunning:
    time.sleep(1)

# When everything done, release the capture and record objects.
del record
del capture
cv2.destroyAllWindows()
