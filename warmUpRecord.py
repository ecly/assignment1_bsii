#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : warmUpRecord.py                                          -->
#<!-- Description: Example of code for recording images using RecordVideo   -->
#<!--            : package                                                  -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 10/02/2017                                               -->
#<!-- Change     : 10/02/2017 - Creation of this script                     -->
#<!-- Review     : 10/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021001 $"

########################################################################
import cv2
import time

from CaptureVideo.CaptureVideo import CaptureVideo
from RecordVideo.RecordVideo   import RecordVideo

########################################################################

# Create a capture video object.
# Tips: You can add a new video capture device or video file with the method
# CaptureVideo.addInputVideo().
capture = CaptureVideo(isDebugging=True)
capture.addInputVideo(0, size=(640, 480), framerate=30.)

# Create a record video object.
# Tips: You can add a new output video file with the method
# RecordVideo.addOutputVideo().
record = RecordVideo(True)
record.addOutputVideo("Outputs/warmUpVideo.mp4", size=(640, 480), framerate=30., isColor=False)

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

    # Convert the input image to grayscale.
    grayscale = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    record.writeFrames(grayscale)

    # Display the resulting frame.
    cv2.imshow("Image", frames)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Start the record thread.
record.stopThread()
while record.IsRunning:
    time.sleep(1)

# When everything done, release the capture and record objects.
del record
del capture
cv2.destroyAllWindows()
