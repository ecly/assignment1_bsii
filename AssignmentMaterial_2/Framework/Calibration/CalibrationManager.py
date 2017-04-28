#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : CalibrationManager.py                                    -->
#<!-- Description: Class used for managing the calibration process          -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 09/04/2015                                               -->
#<!-- Change     : 09/04/2015 - Creation of these classes                   -->
#<!--            : 07/12/2015 - Change the class for the new SIGB Framework -->
#<!-- Review     : 07/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120701 $"

########################################################################
import cv2
import numpy as np

from ClassProperty import ClassProperty
from Pattern       import Pattern

from Framework.VideoCaptureDevices.CaptureManager import CaptureManager

########################################################################
class CalibrationManager(object):
    """CalibrationManager Class is used for managing the calibration process."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance for the calibration manager."""
        if self.__Instance is None:
            self.__Instance = Calibration()
        return self.__Instance

    #----------------------------------------------------------------------#
    #                    ImageManager Class Constructor                    #
    #----------------------------------------------------------------------#
    def __init__(self):
        """This constructor is never used by the system."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Calibration.CalibrationManager object."


########################################################################
class Calibration(object):
    """Calibration Class is used for calibrating the connected cameras."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __path = "./Framework/VideoCaptureDevices/CalibrationData/"

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def IsCalibrated(self):
        """Check if the cameras are calibrated."""
        return self.__isCalibrated

    #----------------------------------------------------------------------#
    #                     Calibration Class Constructor                    #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Calibration Class Constructor."""
        self.__isCalibrated = False
        self.__Pattern = Pattern()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Calibrate(self):
        """Calibrate all connected cameras."""
        # Vectors used by the calibration process.
        objectPoints = []
        imagePoints  = []

        # Get points from the pattern.
        patternPoints = self.__Pattern.CalculatePattern()

        # Define the number of chessboard that you want to use during the calibration process.
        N = 5

        # Number of the camera.
        index = 0

        # Number of detected chessboard.
        j = 0

        # While condition used for calibrating the camera.
        while j < N:
            # Read the current image from a camera.
            image = CaptureManager.Instance.Read()
            h, w  = image.shape[:2]
            chessboard = image.copy()

            # Finds the positions of internal corners of the chessboard.
            corners = self.__Pattern.FindCorners(chessboard)

            # Show the final processed image.
            cv2.imshow("Camera" + str(index) + "_Uncalibrated", chessboard)

            # Checks the keyboard button pressed by the user.
            ch = cv2.waitKey(1)
            if ch == ord("q"):
                break
            elif ch == 32: # Press space key for taking the sample images.
                if corners is not None:
                    j += 1

                    # Saves the detected chessboard.
                    cv2.imwrite(self.__path + "Camera_" + str(index) + "_chessboard" + str(j) + ".png", image)

                    # Add the detected points in the vectors.
                    imagePoints.append(corners.reshape(-1, 2))
                    objectPoints.append(patternPoints)

                    # Long wait for showing to the user the selected chessboard.
                    cv2.waitKey(1000)

        # Close all allocated resources.
        cv2.destroyAllWindows()

        # Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
        if len(imagePoints) > 0:
            # Calibrate a single camera.
            cameraMatrix, distCoeffs = self.CalibrateCamera(index, imagePoints, objectPoints, (w, h))

            # Checks if it is necessary to undistort the image.
            isUndistorting = True

            # While condition used for testing the calibration.
            while True:
                # Read the current image from a camera.
                image = CaptureManager.Instance.Read()

                # Transforms an image to compensate for lens distortion.
                if isUndistorting:
                    image = cv2.undistort(image, cameraMatrix, distCoeffs)

                # Show the final processed image.
                cv2.imshow("Camera" + str(index) + "_Calibrated", image)

                # Checks the keyboard button pressed by the user.
                ch = cv2.waitKey(1)
                if ch == ord("q"):
                    break
                elif ch == ord("u"):
                    isUndistorting = not isUndistorting

            # Define the new calibration status.
            self.__isCalibrated = True

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()

    def CalibrateCamera(self, index, imagePoints, objectPoints, size):
        """Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern."""
        # Output 3x3 floating-point camera matrix and output vector of distortion coefficients.
        cameraMatrix = np.zeros((3, 3))
        distCoeffs   = np.zeros((5, 1))
    
        # Calibrates a single camera.
        _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, size, cameraMatrix, distCoeffs)

        # Save calibration process data.
        np.save(self.__path + "Camera_" + str(index) + "_cameraMatrix", cameraMatrix)
        np.save(self.__path + "Camera_" + str(index) + "_distCoeffs", distCoeffs)
        np.save(self.__path + "Camera_" + str(index) + "_rvecs", rvecs)
        np.save(self.__path + "Camera_" + str(index) + "_tvecs", tvecs)
        np.save(self.__path + "Camera_" + str(index) + "_img_points", imagePoints)
        np.save(self.__path + "Camera_" + str(index) + "_obj_points", objectPoints)

        # Return the final result
        return cameraMatrix, distCoeffs

    def CalibrateStereoCameras(self, leftCorners, rightCorners, objectPoints):
        """Calibrates the stereo camera."""
        pass

    def CrossProductMatrix(self, t):
        """Estimating the skew symmetric matrix."""
        pass

    def EssentialMatrix(self, R, t):
        """Calculate the Essential Matrix."""
        pass

    def FundamentalMatrix(self, K1, K2, E):
        """Calculate the Fundamental Matrix."""
        pass

    def StereoRectify(self, R, t):
        """Computes rectification transforms for each head of a calibrated stereo camera."""
        pass

    def UndistortRectifyMap(self):
        """Computes the undistortion and rectification transformation maps."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Calibration.Calibration object."
