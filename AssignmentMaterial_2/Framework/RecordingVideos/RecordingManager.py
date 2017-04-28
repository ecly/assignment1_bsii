#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : RecordingManager.py                                      -->
#<!-- Description: Class used for managing the recording video process      -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: This class is based on the Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 07/12/2015                                               -->
#<!-- Change     : 07/12/2015 - Creation of these classes                   -->
#<!-- Review     : 07/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120701 $"

########################################################################
import cv2
import numpy as np

import os
from collections import OrderedDict

from ClassProperty import ClassProperty

########################################################################
class RecordingManager(object):
    """RecordingManager Class is used for managing the recording video process."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance for the camera manager."""
        if self.__Instance is None:
            self.__Instance = Recording()
        return self.__Instance

    #----------------------------------------------------------------------#
    #                  RecordingManager Class Constructor                  #
    #----------------------------------------------------------------------#
    def __init__(self):
        """This constructor is never used by the system."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.RecordingVideos.RecordingManager object."

########################################################################
class Recording(object):
    """Recording Class is used for managing the recording video process."""

    #----------------------------------------------------------------------#
    #                      Devices Class Constructor                       #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Devices Class Constructor."""
        # Creates a dictionary for managering multiple cameras.
        self.__videos = OrderedDict()
        self.__fourcc = cv2.VideoWriter_fourcc(*"WMV2")

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def AddVideo(self, index, fps=30.0, size=(640, 480)):
        """Add a new video in the Python dictionary."""
        key = str(index)
        if self.__videos.has_key(key):
            return False

        self.__videos[key] = cv2.VideoWriter(index, self.__fourcc, fps, size)

        return True

    def DelVideo(self, index):
        """Remove a video from the Python dictionary."""
        key = str(index)
        if not self.__videos.has_key(key):
            return False

        self.__videos[key].release()
        del self.__videos[key]

        return True

    def Write(self, images):
        """Writes the next video frame."""
        if type(images).__module__ == np.__name__:
            images = [images]

        for key, image in zip(self.__videos, images):
            self.__videos[key].write(image)

    def Release(self):
        """Closes all videos."""
        for key in self.__videos:
            self.__videos[key].release()
        self.__videos.clear()

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.RecordingVideos.Recording object."
