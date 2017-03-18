#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : ex5.py                                                   -->
#<!-- Description: Script to implement some improvements in your eye        -->
#<!--            : feature detectors                                        -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: Master students MUST change this file                    -->
#<!-- Date       : 01/03/2017                                               -->
#<!-- Change     : 01/03/2017 - Creation of this script                     -->
#<!-- Review     : 03/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017020301 $"

########################################################################
import argparse
import cv2
import time
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

# Define the input video source.
uid = 0
if args["input"] != None:
    if type(args["input"]) is str and args["input"].isdigit():
        uid = int(args["input"])
    else:
        uid = "Inputs/" + args["input"]

# Create the eye feature detector.
detector = EyeFeatureDetector()

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

    # Get the processed image.
    processed = frames.copy()

    # Record the processed frames.
    record.writeFrames(processed)

    # Display the resulting frame.
    cv2.imshow("ex5", processed)
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
