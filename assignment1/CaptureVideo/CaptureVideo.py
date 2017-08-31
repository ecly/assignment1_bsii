#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : CaptureVideo.py                                          -->
#<!-- Description: Class used for managing the input videos used by the     -->
#<!--              Python application                                       -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 03/06/2014                                               -->
#<!-- Change     : 03/06/2014 - Creation of these classes                   -->
#<!--            : 10/02/2017 - Convert the class for IAML course           -->
#<!-- Review     : 10/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021001 $"

########################################################################
from collections  import OrderedDict

from FrameRate  import FrameRate
from InputVideo import InputVideo

########################################################################
class CaptureVideo(object):
    """
    CaptureVideo Class is used for managing several synchronized video
    capture devices.
    """

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def IsDebugging(self):
        """Get a value to check if the class is in debugging mode."""
        return self.__isDebugging

    @property
    def Width(self):
        """Get the current width of input videos."""
        if len(self.__videos) == 0:
            return [0]
        return [self.__videos[key].Width for key in self.__videos]

    @property
    def Height(self):
        """Get the current height of input videos."""
        if len(self.__videos) == 0:
            return [0]
        return [self.__videos[key].Height for key in self.__videos]

    @property
    def FPS(self):
        """Get the input video framerate."""
        if len(self.__videos) == 0:
            return [0]
        return [self.__videos[key].FPS for key in self.__videos]

    @property
    def Count(self):
        """Get the total number of frames of the input videos."""
        if len(self.__videos) == 0:
            return [0]
        return [self.__videos[key].Count for key in self.__videos]

    @property
    def PosFrame(self):
        """Get the current frame position of the input videos."""
        if len(self.__videos) == 0:
            return [0]
        return [self.__videos[key].PosFrame for key in self.__videos]

    @PosFrame.setter
    def PosFrame(self, value):
        """Set a frame position in the input videos."""
        for device in self.__videos.itervalues():
            device.PosFrame = value

    @property
    def Size(self):
        """Get the current input videos size."""
        if len(self.__videos) == 0:
            return [(0, 0)]
        return [self.__videos[key].Size for key in self.__videos]

    @property
    def CurrentFPS(self):
        """Get the current input videos framerate."""
        return self.__FPS

    #----------------------------------------------------------------------#
    #                    CaptureVideo Class Constructor                    #
    #----------------------------------------------------------------------#
    def __init__(self, isDebugging=False):
        """CaptureVideo Class Constructor."""
        if isDebugging:
            print "CaptureVideo: Creating the capture video object."
        
        # Save the isDebugging value in the "self.__isDebugging" property.
        self.__isDebugging = isDebugging

        # Creates a dictionary for managering multiple input videos.
        self.__videos    = OrderedDict()
        self.__framerate = FrameRate()
        self.__FPS = 0

    def __del__(self):
        """CaptureVideo Class Destructor."""
        # Close all connected input videos.
        for key in self.__videos:
            self.__videos[key].releaseVideo()

        self.__videos.clear()

        if self.IsDebugging:
            print "CaptureVideo: Destroying the capture video object."

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def addInputVideo(self, uid, size=(640, 480), framerate=30.):
        """Add a new input video in the Python dictionary."""
        key = str(uid)
        if self.__videos.has_key(key):
            return False

        self.__videos[key] = InputVideo(uid, size, framerate,
                                        self.IsDebugging)

        return True

    def delInputVideo(self, uid, enum):
        """Remove an input video from the Python dictionary."""
        key = str(uid)
        if not self.__videos.has_key(key):
            return False

        self.__videos[key].release()
        del self.__videos[key]

        return True

    def getFrames(self):
        """Grabs, decodes and returns the next video frame."""
        self.__grabFrames()
        frames = self.__retrieveFrames()

        result = []
        retval = True
        for frame in frames:
            retval &= frame[0]
            result.append(frame[1])

        if len(result) == 1:
            return retval, result[0]

        return retval, result

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __grabFrames(self):
        """Grabs the next frame from all input videos."""
        for key in self.__videos:
            self.__videos[key].grabFrames()

    def __retrieveFrames(self):
        """Decodes and returns the grabbed video frame."""
        self.__FPS = self.__framerate.calculateFPS()
        return [self.__videos[key].retrieveFrames() for key in self.__videos]
