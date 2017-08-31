#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : ex6.py                                                   -->
#<!-- Description: Script to select the best threshold for skin color       -->
#<!--            : detection                                                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 27/02/2017                                               -->
#<!-- Change     : 27/02/2017 - Creation of this script                     -->
#<!-- Review     : 01/03/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017030101 $"

########################################################################
import argparse
import cv2
import time
import numpy as np

from SkinColorDetector         import SkinColorDetector
from CaptureVideo.CaptureVideo import CaptureVideo
from RecordVideo.RecordVideo   import RecordVideo

########################################################################
# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, help="Input Video Source")
ap.add_argument("-o", "--output", required=True, help="Output Video Filename")
ap.add_argument("-c", "--cascade", required=True, help="Haar cascade file")
args = vars(ap.parse_args())

# Define the input video source.
uid = 0
if args["input"] != None:
    if type(args["input"]) is str and args["input"].isdigit():
        uid = int(args["input"])
    else:
        uid = "Inputs/" + args["input"]

# Create the skin color detector.
detector = SkinColorDetector(args["cascade"])

# Create a capture video object.
# Tips: You can add a new video capture device or video file with the method
# CaptureVideo.addInputVideo().
capture = CaptureVideo(isDebugging=True)
capture.addInputVideo(uid, size=(640, 480), framerate=30.)

# Get the input image resolution.
_, frames = capture.getFrames()
h, w = frames.shape[:2]

# Create a record video object.
# Tips: You can add a new output video file with the method
# RecordVideo.addOutputVideo().
record = RecordVideo(True)
record.addOutputVideo("Outputs/" + args["output"], size=(w, h),
                      framerate=30., isColor=True)

# Start the record thread.
record.startThread()

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    retval, frames = capture.getFrames()

    # Get the coordinates of the pupil candidates.
    processed = detector.getSkinColor(frames)

    # Get the processed image.
    record.writeFrames(processed[0])

    # Display the resulting frame.
    cv2.imshow("ex6", processed[0])
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
