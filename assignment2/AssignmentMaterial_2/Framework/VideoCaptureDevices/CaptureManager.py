#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : CaptureManager.py                                        -->
#<!-- Description: Class used for managing the cameras connected in the     -->
#<!--              computer                                                 -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: This class is based on the Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 03/06/2014                                               -->
#<!-- Change     : 03/06/2014 - Creation of these classes                   -->
#<!--            : 25/06/2014 - Read the image from Read() method           -->
#<!--            : 04/12/2015 - Camera calibration process                  --
#<!-- Review     : 04/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120401 $"

########################################################################
import cv2
import numpy as np

import os
from collections import OrderedDict

from ClassProperty import ClassProperty
from Enumerations  import *
from FrameRate     import FrameRate
from VideoCapture  import VideoCapture

########################################################################
class CaptureManager(object):
    """CaptureManager Class is used for managing several synchronized video capture devices."""

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
            self.__Instance = Devices()
        return self.__Instance

    #----------------------------------------------------------------------#
    #                   CaptureManager Class Constructor                   #
    #----------------------------------------------------------------------#
    def __init__(self):
        """This constructor is never used by the system."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.VideoCaptureDevices.CaptureManager object."

########################################################################
class Devices(object):
    """Devices Class is used for managing the several synchronized cameras."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __path = "./Framework/VideoCaptureDevices/CalibrationData/"

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Width(self):
        """Get the current width of captured images."""
        if len(self.__devices) == 0:
            return 0
        return self.__devices.itervalues().next().Width

    @property
    def Height(self):
        """Get the current height of captured images."""
        if len(self.__devices) == 0:
            return 0
        return self.__devices.itervalues().next().Height

    @property
    def FPS(self):
        """Get the current frames per second of the cameras"""
        return self.__FPS

    @property
    def Size(self):
        """Get the current size of captured images."""
        if len(self.__devices) == 0:
            return (0, 0)
        return self.__devices.itervalues().next().Size

    @property
    def Parameters(self):
        """Get the camera parameters."""
        matrices = []

        for index in self.__devices:
            matrices.append(self.__devices[index].Parameters)
    
        if len(matrices) == 1:
            return matrices[0]
        return matrices

    #----------------------------------------------------------------------#
    #                      Devices Class Constructor                       #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Devices Class Constructor."""
        # Creates a dictionary for managering multiple cameras.
        self.__devices     = OrderedDict()
        self.__framerate   = FrameRate()
        self.__FPS = 0

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def AddCamera(self, index, enum):
        """Add a new camera in the Python dictionary."""
        key = str(index)
        if self.__devices.has_key(key):
            return False

        self.__devices[key] = VideoCapture(index)
        if enum is CAMERA_VIDEOCAPTURE_320X240:
            self.__devices[key].Size = (320, 240)
        elif enum is CAMERA_VIDEOCAPTURE_320X240_15FPS:
            self.__devices[key].Size = (320, 240)
            self.__devices[key].FPS  = 15
        elif enum is CAMERA_VIDEOCAPTURE_320X240_30FPS:
            self.__devices[key].Size = (320, 240)
            self.__devices[key].FPS  = 30
        elif enum is CAMERA_VIDEOCAPTURE_640X480:
            self.__devices[key].Size = (640, 480)
        elif enum is CAMERA_VIDEOCAPTURE_640X480_15FPS:
            self.__devices[key].Size = (640, 480)
            self.__devices[key].FPS  = 15
        else:
            self.__devices[key].Size = (640, 480)
            self.__devices[key].FPS  = 30        

        self.__Calibration(key)

        return True

    def DelCamera(self, index):
        """Remove a camera from the Python dictionary."""
        key = str(index)
        if not self.__devices.has_key(key):
            return False

        self.__devices[key].Release()
        del self.__devices[key]

        return True

    def Read(self):
        """Grabs, decodes and returns the next video frame."""
        self.__Grab()
        images = self.__Retrieve()

        if len(images) == 1:
            return images[0]

        return images

    def Release(self):
        """Closes all cameras."""
        for key in self.__devices:
            self.__devices[key].Release()
        self.__devices.clear()

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __Grab(self):
        """Grabs the next frame from all cameras."""
        for key in self.__devices:
            self.__devices[key].Grab()

    def __Retrieve(self):
        """Decodes and returns the grabbed video frame."""
        self.__FPS = self.__framerate.CalculateFPS()
        return [self.__devices[key].Retrieve() for key in self.__devices]

    def __Calibration(self, key):
        """Import the calibration files."""
        filename = self.__path + "Camera_" + str(key) + "_cameraMatrix.npy"
        if os.path.isfile(filename):
            K = np.load(filename)
            self.__devices[key].Parameters.K = K
            filename = self.__path + "Camera_" + str(key) + "_rvecs.npy"
            if os.path.isfile(filename):
                R = np.load(filename)
                self.__devices[key].Parameters.R = R
                r  = cv2.Rodrigues(R[0])[0]
                filename = self.__path + "Camera_" + str(key) + "_tvecs.npy"
                if os.path.isfile(filename):
                    t = np.load(filename)
                    self.__devices[key].Parameters.t = t
                    RT = np.hstack((r, t[0]))
                    self.__devices[key].Parameters.P = np.dot(K, RT)

        filename = self.__path + "Camera_" + str(key) + "_distCoeffs.npy"
        if os.path.isfile(filename):
            self.__devices[key].Parameters.DistCoeffs = np.load(filename)

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.VideoCaptureDevices.Devices object."
