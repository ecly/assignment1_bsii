#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : RecordVideo.py                                           -->
#<!-- Description: Class used for managing the recording video process      -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 07/12/2015                                               -->
#<!-- Change     : 07/12/2015 - Creation of this class                      -->
#<!--            : 10/02/2017 - Convert the class for IAML course           -->
#<!-- Review     : 10/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021001 $"

########################################################################
import cv2
import copy
import thread
import numpy as np

from collections import deque
from collections import OrderedDict

########################################################################
class RecordVideo(object):
    """RecordVideo Class is used for managing the recording video process."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def IsDebugging(self):
        """Get a value to check if the class is in debugging mode."""
        return self.__isDebugging

    @property
    def IsRunning(self):
        """Get a value to check if the main thread is running."""
        return self.__isRunning

    #----------------------------------------------------------------------#
    #                    RecordVideo Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self, isDebugging=False):
        """RecordVideo Class Constructor."""
        if isDebugging:
            print "RecordVideo: Creating the record video object."
        
        # Save the isDebugging value in the "self._isDebugging" property.
        self.__isDebugging = isDebugging

        # Creates a dictionary for managering multiple cameras.
        self.__videos = OrderedDict()
        self.__images = deque()

        # Codec manager.
        if cv2.__version__[0] == "2":
            self.__fourcc = cv2.cv.FOURCC("m", "p", "4", "v")
        else:
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Thread manager.
        self.__isRecording = False
        self.__isRunning   = False

    def __del__(self):
        """RecordVideo Class Constructor."""
        if self.IsDebugging:
            print "RecordVideo: Destroying the record video object."

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def addOutputVideo(self, uid, size=(640, 480), framerate=30., isColor=True):
        """Add a new video in the Python dictionary."""
        key = str(uid)
        if self.__videos.has_key(key):
            if self.IsDebugging:
                print "RecordVideo: There is an output video with the same filename (%s)." % (key)
            return False

        if (self.__isRecording or len(self.__images) > 0):
            if self.IsDebugging:
                print "RecordVideo: You cannot add a new output vide if the record thread is running."
            return False

        if self.IsDebugging:
            print "RecordVideo: Adding a new output video to the record process."

        self.__videos[key] = cv2.VideoWriter(uid, self.__fourcc, framerate, size, isColor)

        return True

    def delOutputVideo(self, uid):
        """Remove a video from the Python dictionary."""
        key = str(uid)
        if not self.__videos.has_key(key):
            if self.IsDebugging:
                print "RecordVideo: There isn't any output video with the filename (%s)." % (key)
            return False

        if self.IsDebugging:
            print "RecordVideo: Releasing the output video with the filename (%s)." % (key)

        self.__videos[key].release()
        del self.__videos[key]

        return True

    def writeFrames(self, images):
        """Write the next video frame."""
        if type(images).__module__ == np.__name__:
            images = [images]

        if len(images) != len(self.__videos):
            if self.IsDebugging:
                print "RecordVideo: Number of images is different of output size."
            return False

        self.__images.append(copy.copy(images))

        return True

    def startThread(self):
        """Check and start the record output video thread."""
        if len(self.__videos) == 0:
            if self.IsDebugging:
                print "RecordVideo: There is no valid output video file."
            return False

        if self.__isRecording:
            if self.IsDebugging:
                print "RecordVideo: There is an active thread."
            return False

        thread.start_new_thread(self.__executeThread, ())

        return True

    def stopThread(self):
        """Check and stop the record output video thread."""
        if not self.__isRecording:
            if self.IsDebugging:
                print "RecordVideo: There is no active thread."
            return False

        self.__isRecording = False

        return True

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __executeThread(self):
        """Main thread of recording process."""
        if self.IsDebugging:
            print "RecordVideo: Recording video(s)."

        # Change the recording status.
        self.__isRecording = True
        self.__isRunning   = True

        # Main loop.
        cont = 0
        while self.__isRecording or len(self.__images) > 0:
            # Record the next frame.
            if len(self.__images) > 0:
                cont += 1

                if self.IsDebugging:
                    print "RecordVideo: Recording frame %d of %d." % (cont, len(
                        self.__images) + cont - 1)

                images = self.__images.popleft()

                for key, image in zip(self.__videos, images):
                    self.__videos[key].write(image)

        # Close the writers.
        for key in self.__videos:
            self.__videos[key].release()
        self.__videos.clear()
        self.__fps = 0

        self.__isRunning = False
        if self.IsDebugging:
            print "RecordVideo: Done."
