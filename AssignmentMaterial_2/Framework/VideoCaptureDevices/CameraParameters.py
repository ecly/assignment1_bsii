#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : CameraParameters.py                                      -->
#<!-- Description: Class used for managing the cameras parameters           -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 07/04/2015                                               -->
#<!-- Change     : 07/04/2015 - Creation of these classes                   -->
#<!--            : 07/12/2015 - Adapter for the new SIGB Framework          -->
#<!-- Review     : 07/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120701 $"

########################################################################
import numpy as np
from scipy.linalg import qr

########################################################################
class CamerasParameters(object):
    """CamerasParameters Class is used for managing the cameras parameters."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def P(self):
        """Get the projection matrix."""
        return self.__P

    @P.setter
    def P(self, value):
        """Set the projection matrix."""
        self.__P = value

    @property
    def K(self):
        """Get the camera matrix."""
        return self.__K

    @K.setter
    def K(self, value):
        """Set the camera matrix."""
        self.__K = value

    @property
    def R(self):
        """Get the rotation matrix."""
        return self.__R

    @R.setter
    def R(self, value):
        """Set the rotation matrix."""
        self.__R = value

    @property
    def t(self):
        """Get the translation matrix."""
        return self.__t

    @t.setter
    def t(self, value):
        """Set the translation matrix."""
        self.__t = value

    @property
    def DistCoeffs(self):
        """Get the camera distortion coefficients."""
        return self.__DistCoeffs

    @DistCoeffs.setter
    def DistCoeffs(self, value):
        """Set the camera distortion coefficients."""
        self.__DistCoeffs = value

    #----------------------------------------------------------------------#
    #                  CamerasParameters Class Constructor                 #
    #----------------------------------------------------------------------#
    def __init__(self):
        """CamerasParameters Class Constructor."""
        self.Clear()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Center(self):
        """Compute and return the camera center."""
        _, R, t = self.__Factor()
        c = -np.dot(R.T, t)
        return c

    def Factor(self):
        """Factorize the camera matrix into K, R, t as P = K[R|t]."""
        P = np.matrix(self.P)

        if P.max() == 0:
            return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 1))

        K, R = self.__RQ(P[:, :3])
        t = np.diag(np.sign(np.diag(K)))

        K = np.dot(K, t)
        R = np.dot(t, R)
        t = np.dot(np.linalg.inv(K), P[:, 3])

        return K, R, t

    def Project(self, point):
        """Project the input point based on a camera matrix."""
        if point.shape[0] != 4:
            point = np.vstack((point, np.ones((1, point.shape[1]))))

        point = np.dot(self.P, point)
        for i in range(3):
            point[i] /= point[2]

        return np.array(point.T)

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __RQ(self, matrix):
        """Estimate the factor first 3*3 part."""
        Q, R = qr(np.flipud(matrix).T)
        R = np.flipud(R.T)
        Q = Q.T

        return R[:, ::-1], Q[::-1, :]

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def Clear(self):
        """Define the default values for all class attributes."""
        self.P = np.zeros((3, 4))
        self.K = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.t = np.zeros((3, 1))
        self.DistCoeffs = np.zeros((1, 5))

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.VideoCaptureDevices.CameraParameters object."
