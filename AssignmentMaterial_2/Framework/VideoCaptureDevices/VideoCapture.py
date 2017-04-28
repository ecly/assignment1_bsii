#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : VideoCapture.py                                          -->
#<!-- Description: Class for video capturing from video files, image        -->
#<!--            : sequences or cameras                                     -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 03/06/2014                                               -->
#<!-- Change     : 03/06/2014 - Creation of this class                      -->
#<!--            : 05/12/2015 - Camera calibration                          -->
#<!-- Review     : 05/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120501 $"

########################################################################
import cv2

from AbstractDevice   import AbstractDevice
from CameraParameters import CamerasParameters

########################################################################
class ExceptionError(RuntimeError):
    """ExceptionError class raises when an error is detected that doesn't fall in any of the other exception categories.
       The associated value is a string indicating what precisely went wrong."""
    pass

########################################################################
class VideoCapture(AbstractDevice):
    """Class for video capturing from video files, image sequences or cameras."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Width(self):
        """Get the current width of captured images."""
        return int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    @Width.setter
    def Width(self, value):
        """Set a new width value to captured images."""
        self.__camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(value))

    @property
    def Height(self):
        """Get the current height of captured images."""
        return int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @Height.setter
    def Height(self, value):
        """Set a new height value to captured images."""
        self.__camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(value))

    @property
    def FPS(self):
        """Get the current frames per second of the cameras"""
        return int(self.__camera.get(cv2.CAP_PROP_FPS))

    @FPS.setter
    def FPS(self, value):
        """Set a new frames per second of the cameras."""
        self.__camera.set(cv2.CAP_PROP_FPS, int(value))

    @property
    def Size(self):
        """Get the current size of captured images."""
        return (self.Width, self.Height)

    @Size.setter
    def Size(self, value):
        """Set a new size to captured images."""
        self.Width, self.Height = map(int, value)

    @property
    def Parameters(self):
        """Get the camera parameters."""
        return self.__Parameters

    @Parameters.setter
    def Parameters(self, value):
        """Set the camera parameters."""
        self.__Parameters = value

    #----------------------------------------------------------------------#
    #             VideoCapture Class Constructor and Destructor            #
    #----------------------------------------------------------------------#
    def __init__(self, index):
        """VideoCapture Class Constructor"""
        # Save the index value in the "self._index" property.
        self._index = index

        # Create the camera calibration parameters.
        self.Parameters = CamerasParameters()

        # Create an instance of a video capture device based on index argument.
        self.__camera = cv2.VideoCapture(index)
        self.__camera.set(cv2.CAP_PROP_CONVERT_RGB, True)

        # Check if the video capture device has been initialized already using "isOpened()" function.
        # Otherwise, throw an exception with a message about a problem in the camera initialization process.
        if not self.__camera.isOpened():
            raise ExceptionError("There was a failed during initialization process for camera index {}".format(self.Index))

    #----------------------------------------------------------------------#
    #                        Abstract Class Methods                        #
    #----------------------------------------------------------------------#
    def Grab(self):
        """Grabs the next frame from video file or capturing device."""
        # Case there is any problem during this process, throw an exception with a message about the grab problem.
        if not self.__camera.grab():
            if type(self.Index) is str:
                self.__camera.set(cv2.CAP_PROP_POS_FRAMES, 1)
                return
            raise ExceptionError("There was a failed during grab process for camera index {}".format(self.Index))

    def Retrieve(self):
        """Decodes and returns the grabbed video frame."""
        # Get the grabbed image.
        retval, image = self.__camera.retrieve()

        # Case there is any problem during this process, throw an exception with a message about the retrieve problem.
        if not retval:
            raise ExceptionError("There was a failed during retrieve process for camera index {}".format(self.Index))

        # Convert the input image to RGB color.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Return the grabbed image.
        return image

    def Release(self):
        """Closes video file or capturing device."""
        self.__camera.release()

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.VideoCaptureDevices.VideoCapture object of webcam ID %s." % str(self.Index)
