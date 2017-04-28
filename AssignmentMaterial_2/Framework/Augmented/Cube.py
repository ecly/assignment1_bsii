#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Cube.py                                                  -->
#<!-- Description: Class used for creating an augmented cube                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 18/04/2015                                               -->
#<!-- Change     : 18/04/2015 - Creation of these classes                   -->
#<!--            : 05/12/2015 - Update the class for the new assignment     -->
#<!-- Review     : 05/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015051201 $'

########################################################################
import cv2
import numpy as np
import math

from Framework.VideoCaptureDevices.CaptureManager import CaptureManager
from Framework.ImageProcessing.ImageManager  import ImageManager

########################################################################
class Cube(object):
    """Cube Class is used creating an augmented cube."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Object(self):
        """Get an augmented cube."""
        return self.__Object

    @property
    def CoordinateSystem(self):
        """Get the augmented coordinate system."""
        return self.__CoordinateSystem

    #----------------------------------------------------------------------#
    #                      Augmented Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Augmented Class Constructor."""
        # Creates the augmented objects used by this class.
        self.__CreateObjects()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def PoseEstimationMethod1(self, image, corners, homographyPoints, calibrationPoints, projectionMatrix, cameraMatrix):
        """This method uses the homography between two views for finding the extrinsic parameters (R|t) of the camera in the current view."""
        pass

    def PoseEstimationMethod2(self, corners, patternPoints, cameraMatrix, distCoeffs):
        """This function uses the chessboard pattern for finding the extrinsic parameters (R|T) of the camera in the current view."""
        pass

    def DrawCoordinateSystem(self, image):
        """Draw the coordinate axes attached to the chessboard pattern."""
        pass

    def DrawAugmentedCube(self, image):
        """Draw a cube attached to the chessboard pattern."""
        pass

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __CreateObjects(self):
        """Defines the points of the augmented objects based on calibration patterns."""
        # Creates an augmented cube.
        self.__Object = np.float32([[3, 1,  0], [3, 4,  0], [6, 4,  0], [6, 1,  0],
                                    [3, 1, -3], [3, 4, -3], [6, 4, -3], [6, 1, -3]]).T
        # Creates the coordinate system.
        self.__CoordinateSystem = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).reshape(-1, 3)

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Augumented.Cube object."
