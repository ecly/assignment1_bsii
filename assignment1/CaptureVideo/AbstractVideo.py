#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : AbstractVideo.py                                         -->
#<!-- Description: Abstract class used for abstracting all input videos     -->
#<!--              used by the Python application                           -->
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
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

########################################################################
class AbstractVideo(object):
    """
    Main class for abstracting all input videos used by the Python
    application.
    """

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __metaclass__ = ABCMeta

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def UID(self):
        """Get the uid used for managering the input video."""
        return self._uid

    @property
    def IsDebugging(self):
        """Get a value to check if the class is in debugging mode."""
        return self._isDebugging

    @abstractproperty
    def Width(self):
        """Get the current width of input videos."""

    @abstractproperty
    def Height(self):
        """Get the current height of input videos."""

    @abstractproperty
    def FPS(self):
        """Get the input video framerate."""

    @abstractproperty
    def Count(self):
        """Get the total number of frames of the input videos."""

    @abstractproperty
    def PosFrame(self):
        """Get the current frame position of the input videos."""

    @PosFrame.setter
    def PosFrame(self, value):
        """Set a frame position in an input video."""

    @abstractproperty
    def Size(self):
        """Get the current input videos size."""

    #----------------------------------------------------------------------#
    #                        Abstract Class Methods                        #
    #----------------------------------------------------------------------#
    @abstractmethod
    def grabFrames(self):
        """Grabs the next frame from video file or capturing device."""

    @abstractmethod
    def retrieveFrames(self):
        """Decodes and returns the grabbed video frame."""

    @abstractmethod
    def releaseVideo(self):
        """Closes video file or capturing device."""
