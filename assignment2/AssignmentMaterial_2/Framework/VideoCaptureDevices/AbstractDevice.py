#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : AbstractDevice.py                                        -->
#<!-- Description: Abstract class used for abstracting all cameras used     -->
#<!--              by SIGB framework                                        -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 03/06/2014                                               -->
#<!-- Change     : 03/06/2014 - Creation of this class                      -->
#<!-- Review     : 05/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120501 $"

########################################################################
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

########################################################################
class AbstractDevice(object):
    """Main class for abstracting all cameras used by SIGB framework."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __metaclass__ = ABCMeta

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Index(self):
        """Get the index used for managering the video camera device."""
        return self._index

    @abstractproperty
    def Width(self):
        """Get the current width of captured images."""

    @Width.setter
    def Width(self, value):
        """Set a new width value to captured images."""

    @abstractproperty
    def Height(self):
        """Get the current height of captured images."""

    @Height.setter
    def Height(self, value):
        """Set a new height value to captured images."""

    @abstractproperty
    def FPS(self):
        """Get the current frames per second of the cameras"""

    @FPS.setter
    def FPS(self, value):
        """Set a new frames per second of the cameras."""

    @abstractproperty
    def Size(self):
        """Get the current size of captured images."""

    @Size.setter
    def Size(self, value):
        """Set a new size to captured images."""

    @abstractproperty
    def Parameters(self):
        """Get the camera parameters."""

    @Parameters.setter
    def Parameters(self, value):
        """Set the camera parameters."""

    #----------------------------------------------------------------------#
    #                        Abstract Class Methods                        #
    #----------------------------------------------------------------------#
    @abstractmethod
    def Grab(self):
        """Grabs the next frame from video file or capturing device."""

    @abstractmethod
    def Retrieve(self):
        """Decodes and returns the grabbed video frame."""

    @abstractmethod
    def Release(self):
        """Closes video file or capturing device."""

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.VideoCaptureDevices.AbstractDevice object at device ID %s." % str(self.Index)
