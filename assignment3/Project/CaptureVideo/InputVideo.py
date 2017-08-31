#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : InputVideo.py                                            -->
#<!-- Description: Class for video capturing from video files, image        -->
#<!--            : sequences or cameras                                     -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 03/06/2014                                               -->
#<!-- Change     : 03/06/2014 - Creation of this class                      -->
#<!--            : 10/02/2017 - Convert the class for IAML course           -->
#<!-- Review     : 10/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021001 $"

########################################################################
import cv2
import numpy as np

from AbstractVideo import AbstractVideo

########################################################################
class InputVideo(AbstractVideo):
    """
    Class for video capturing from video files, image sequences or cameras.
    """

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Width(self):
        """Get the current width of input videos."""
        return int(self.__video.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def Height(self):
        """Get the current height of input videos."""
        return int(self.__video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def FPS(self):
        """Get the input video framerate."""
        return int(self.__video.get(cv2.CAP_PROP_FPS))

    @property
    def Count(self):
        """Get the total number of frames of the input videos."""
        return int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def PosFrame(self):
        """Get the current frame position of the input videos."""
        return int(self.__video.get(cv2.CAP_PROP_POS_FRAMES))

    @PosFrame.setter
    def PosFrame(self, value):
        """Set a frame position in an input video."""
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, value)

    @property
    def Size(self):
        """Get the current input videos size."""
        return (self.Width, self.Height)

    #----------------------------------------------------------------------#
    #              InputVideo Class Constructor and Destructor             #
    #----------------------------------------------------------------------#
    def __init__(self, uid="0", size=(640, 480), framerate=30.,
                 isDebugging=False):
        """InputVideo Class Constructor."""
        if isDebugging:
            print "InputVideo: Creating the OpenCV video object (UID=%s)." % (uid)

        # Save the uid value in the "self._uid" property.
        self._uid = uid

        # Save the isDebugging value in the "self._isDebugging" property.
        self._isDebugging = isDebugging

        # Create an instance of a video capture device or a video file based
        # on the uid argument.
        self.__video = cv2.VideoCapture(uid)

        # Change the setting of a video capture device.
        if type(uid) is int or (type(uid) is str and uid.isdigit()):
            self.__video.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.__video.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.__video.set(cv2.CAP_PROP_FPS, framerate)

        # Last captured image.
        self.__lastImage = np.zeros((size[1], size[0], 3), np.uint8)

    def __del__(self):
        """InputVideo Class Destructor."""
        if self.IsDebugging:
            print "InputVideo: Destroying the OpenCV video object (UID=%s)." % (self.UID)

    #----------------------------------------------------------------------#
    #                        Abstract Class Methods                        #
    #----------------------------------------------------------------------#
    def grabFrames(self):
        """Grabs the next frame from video file or capturing device."""
        # Case there is any problem during this process, throw an exception
        # with a message about the grab problem.
        self.__video.grab()

    def retrieveFrames(self):
        """Decodes and returns the grabbed video frame."""
        # Get the grabbed image.
        retval, image = self.__video.retrieve()

        # Case there is any problem during this process, get the last image.
        if not retval:
            image = self.__lastImage
        else:
            self.__lastImage = image

        # Return the grabbed image.
        return retval, image

    def releaseVideo(self):
        """Closes video file or capturing device."""
        if self.IsDebugging:
            print "InputVideo: Releasing the OpenCV video object (UID=%s)." % (self.UID)

        self.__video.release()
