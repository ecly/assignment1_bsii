#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : SIGBTools.py                                             -->
#<!-- Description: Main class of this project                               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/02/2016                                               -->
#<!-- Change     : 24/02/2016 - Creation of this class                      -->
#<!--            : 29/03/2016 - Add new features for Assignment #02         -->
#<!-- Review     : 29/03/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016032901 $"

########################################################################
import math
import numpy as np
import os
import pickle

from Framework.Augmented.AugmentedManager         import AugmentedManager
from Framework.Calibration.CalibrationManager     import CalibrationManager
from Framework.ImageProcessing.ImageManager       import ImageManager
from Framework.ImageProcessing.RegionProps        import RegionProps as Props
from Framework.ImageProcessing.ROISelector        import ROISelector as Selector
from Framework.RecordingVideos.RecordingManager   import RecordingManager
from Framework.VideoCaptureDevices.CaptureManager import CaptureManager
from Framework.VideoCaptureDevices.Enumerations   import *

########################################################################

#----------------------------------------------------------------------#
#                         RegionProps Methods                          #
#----------------------------------------------------------------------#
def RegionProps():
    """This class is used for getting descriptors of contour-based connected components.

        The main method to use is: CalcContourProperties(contour, properties=[]):
        contour: a contours found through cv2.findContours()
        properties: list of strings specifying which properties should be calculated and returned

        The following properties can be specified:

        Area: Area within the contour - float
        Boundingbox: Bounding box around contour - 4 tuple (topleft.x, topleft.y, width, height)
        Length: Length of the contour
        Centroid: The center of contour - (x, y)
        Moments: Dictionary of moments
        Perimiter: Permiter of the contour - equivalent to the length
        Equivdiameter: sqrt(4 * Area / pi)
        Extend: Ratio of the area and the area of the bounding box. Expresses how spread out the contour is
        Convexhull: Calculates the convex hull of the contour points
        IsConvex: boolean value specifying if the set of contour points is convex

        Returns: Dictionary with key equal to the property name

        Example: 
             image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
             goodContours = []
             for cnt in contours:
                vals = props.CalcContourProperties(cnt, ["Area", "Length", "Centroid", "Extend", "ConvexHull"])
                if vals["Area"] > 100 and vals["Area"] < 200:
                   goodContours.append(cnt)
    """
    return Props()

#----------------------------------------------------------------------#
#                         ROISelector Methods                          #
#----------------------------------------------------------------------#
def ROISelector(image):
    """This class returns the corners of the selected area as: [(UpperLeftcorner), (LowerRightCorner)].
       Use the Right Mouse Button to set upper left hand corner and the Left Mouse Button to set the lower right corner.
        Click on the image to set the area
        Keys:
            Enter/SPACE - OK
            ESC/q       - Exit (Cancel)"""
    return Selector(image)

#----------------------------------------------------------------------#
#                       RecordingVideos Methods                        #
#----------------------------------------------------------------------#
def RecordingVideos(filepath, fps=30.0, size=(640, 480)):
    """
    RecordingVideos(filepath, size) -> True or False

    Class for recording videos. The class provides access to OpenCV for recording multiple videos and image sequences.
    Returns: A boolean value inform if the video capture device was correct connected.
    Parameters: filepath: the complete file path of the recorded video,
                size: a tuple with the output video size [Format: (X, Y), where X is the video width and Y is the video height].

    Usage: SIGBTools.RecordingVideos("C:\output.wmv", (640, 480))
    """
    return RecordingManager.Instance.AddVideo(filepath, fps=fps, size=size)

def write(images):
    """
    write(images)

    Writes the next video frame.
    Returns: This method does not return anything.
    Parameters: images: the input images.

    Usage: SIGBTools.write(images)
    """
    return RecordingManager.Instance.Write(images)

def close():
    """
    close()

    Closes video writer.
    Returns: This method does not return anything.
    Parameters: This method does not have any parameter.

    Usage: SIGBTools.close()
    """
    RecordingManager.Instance.Release()

#----------------------------------------------------------------------#
#                          Augmented Methods                           #
#----------------------------------------------------------------------#
def CalculatePattern():
    """
    CalculatePattern() -> objectPoints

    Creates a standard vectors of the calibration pattern points.
    Returns: objectPoints: a vector of points from a chessboard pattern.
    Parameters: This method does not have any parameter.

    Usage: patternPoints = SIGBTools.CalculatePattern()
    """
    return AugmentedManager.Instance.Pattern.CalculatePattern()

def DrawCoordinateSystem(image):
    """
    DrawCoordinateSystem(image)

    Draw the coordinate axes attached to the chessboard pattern.
    Returns: This method does not return anything.
    Parameters: image: input image which the coordinate system will be drawn.

    Usage: SIGBTools.DrawCoordinateSystem(image)
    """
    AugmentedManager.Instance.Cube.DrawCoordinateSystem(image)

def DrawAugmentedCube(image):
    """
    DrawAugmentedCube(image)

    Draw a cube attached to the chessboard pattern.
    Returns: This method does not return anything.
    Parameters: image: input image which the augmented cube will be drawn.

    Usage: SIGBTools.DrawAugmentedCube(image)
    """
    AugmentedManager.Instance.Cube.DrawAugmentedCube(image)

def FindCorners(image, isDrawed=True):
    """
    FindCorners(image, isDrawed=True) -> corners

    The function attempts to determine whether the input image is a view of the chessboard pattern and locate the internal chessboard corners.
    Returns: corners: an array of detected corners.
    Parameters: image: source chessboard view. It must be an 8-bit grayscale or color image.
                isDrawed: a boolean value informe that it is necessary to draw over the detected chessboard.

    Usage: corners = SIGBTools.FindCorners(image)
           corners = SIGBTools.FindCorners(image, isDrawed=True)
    """
    return AugmentedManager.Instance.Pattern.FindCorners(image)

def PoseEstimationMethod1(image, corners, homographyPoints, calibrationPoints, P, K):
    """
    PoseEstimationMethod1(image, corners, homographyPoints, calibrationPoints, P, K) -> P2

    This method uses the homography between two views for finding the extrinsic parameters (R|t) of the camera in the current view.
    Returns: P2: an updated 3x4 floating-point projection matrix.
    Parameters: image: input image.
                corners: corners from the detected chessboard.
                homographyPoints: points used to estimate a homography to the detected chessboard.
                calibrationPoints: points used during the camera calibration process.
                P: a 3x4 floating-point projection matrix.
                K: a 3x3 floating-point camera matrix.

    Usage: P = SIGBTools.PoseEstimationMethod1(image, corners, homographyPoints, calibrationPoints, P, K)
    """
    return AugmentedManager.Instance.Cube.PoseEstimationMethod1(image, corners, homographyPoints, calibrationPoints, P, K)

def PoseEstimationMethod2(corners, patternPoints, K, distCoeffs):
    """
    PoseEstimationMethod2(corners, patternPoints, K, distCoeffs) -> P2

    This function uses the chessboard pattern for finding the extrinsic parameters (R|T) of the camera in the current view.
    Returns: P2: an updated 3x4 floating-point projection matrix.
    Parameters: corners: corners from the detected chessboard.
                patternPoints: points from the detected chessboard.
                K: a 3x3 floating-point camera matrix.
                distCoeffs: vector of distortion coefficients of 4, 5, or 8 elements.

    Usage: P = PoseEstimationMethod2(corners, patternPoints, K, distCoeffs)
    """
    return AugmentedManager.Instance.Cube.PoseEstimationMethod2(corners, patternPoints, K, distCoeffs)

#----------------------------------------------------------------------#
#                         Geometrical Methods                          #
#----------------------------------------------------------------------#
def GetCircleSamples(center=(0, 0), radius=1, numPoints=30):
    """
    GetCircleSamples(center=(0, 0), radius=1, numPoints=30) -> (x, y, d_x, d_y)

    Samples a circle with center center = (x, y), radius = 1 and in numPoints on the circle.
    Returns: an array of a tuple containing the points (x, y) on the circle and the curve gradient in the point (d_x, d_y).
             Notice the gradient (d_x, d_y) has unit length.
    Parameters: center: (x, y) circle center.
                radius: circle radius.
                numPoints: number of points in the circle.

    Usage: P = SIGBTools.GetCircleSamples((100, 100), 40, 20)
    """
    s = np.linspace(0, 2 * math.pi, numPoints)
    P = [ (radius * np.cos(t) + center[0], radius * np.sin(t) + center[1], np.cos(t), np.sin(t)) for t in s ]

    return P

def GetLineCoordinates(p1, p2):
    """
    GetLineCoordinates(p1, p2) -> coordinates

    Get integer coordinates between p1 and p2 using Bresenhams algorithm.
    Returns: a coordinate of I along the line from p1 to p2.
    Parameters: p1: A cartesian coordinate.
                p2: A cartesian coordinate.

    Usage: coordinates = SIGBTools.GetLineCoordinates((x1,y1),(x2,y2))
    """
    (x1, y1) = p1
    x1 = int(x1)
    y1 = int(y1)

    (x2, y2) = p2
    x2 = int(x2)
    y2 = int(y2)

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True

    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)

    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1

    for x in range(x1, x2 + 1):
        if issteep:
            points.append([y, x])
        else:
            points.append([x, y])
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax

    # Reverse the list if the coordinates were reversed.
    if rev:
        points.reverse()

    retPoints = np.array(points)
    X = retPoints[:, 0]
    Y = retPoints[:, 1]

    return retPoints

#----------------------------------------------------------------------#
#                       ImageProcessing Methods                        #
#----------------------------------------------------------------------#
def DetectPlaneObject(image, minSize=1000):
    """
    DetectPlaneObject(image, minSize=1000) -> squares

    A simple attempt to detect rectangular color regions in the image.
    Returns: The squares of detected object.
    Parameters: image: input image
                minSize: the minimum size of detected object.

    Usage: squares = SIGBTools.DetectPlaneObject(image)
           squares = SIGBTools.DetectPlaneObject(image, 2000)
    """
    return ImageManager.Instance.DetectPlaneObject(image)

def FrameTrackingData2BoxData(data):
    """
    FrameTrackingData2BoxData(data) -> rectangle

    Convert a row of points into tuple of points for each rectangle.
    Returns: A rectangle with a tuple of points.
    Parameters: data: a row of points

    Usage: SIGBTools.FrameTrackingData2BoxData(data)
    """
    return ImageManager.Instance.FrameTrackingData2BoxData(data)

def GetHomographyTG(image, homography, texture, scale):
    """
    GetHomographyTG(image, homography, texture, scale) -> 3x3 homography matrix

    Calculate the homography H_t^g from T to G via the homography M.
    Returns: A 3x3 homography matrix.
    Parameters: image: input image,
                homography: the homography M.
                texture: the texture to be applied in the input image.
                scale: an integer value to define the scalling used by the texture image.

    Usage: Htg = SIGBTools.GetHomographyTG(image, homography, texture, scale)
    """
    return ImageManager.Instance.GetHomographyTG(image, homography, texture, scale)

def GetHomographyFromMouse(image1, image2, N=4):
    """
    GetHomographyFromMouse(image1, image2, N=4) -> 3x3 homography matrix

    Method for selecting corresponding points and calculating the homography matrix between two images.
    Returns: A 3x3 homography matrix.
    Parameters: image1: first input image,
                image2: second input image,
                N=4: number of corresponding points for calculating the homography matrix. (N >= 4)

    Usage: SIGBTools.GetHomographyFromMouse(image1, image2)
           SIGBTools.GetHomographyFromMouse(image1, image2, 4)
    """
    return ImageManager.Instance.GetHomographyFromMouse(image1, image2, N)

def TextureMoving(image, gridPoints, texture):
    """
    def TextureMoving(image, gridPoints, texture) -> image
    
    Develop a way of solving the problem so that the texture is mapped even during rotations.
    Returns: A processed image.
    Parameters: image: the input image,
                gridPoints: points of a rectangle,
                texture: the texture to be applied in the input image.

    Usage: SIGBTools.TextureMoving(image, gridPoints, texture)
    """
    return ImageManager.Instance.TextureMoving(image, gridPoints, texture)

#----------------------------------------------------------------------#
#                         Calibration Methods                          #
#----------------------------------------------------------------------#
def calibrate():
    """
    calibrate()
    
    Method used for calibrating all connected cameras.
    Returns: This method does not return anything.
    Parameters: This method does not have any parameter.

    Usage: SIGBTools.calibrate()
    """
    CalibrationManager.Instance.Calibrate()

def GetCameraParameters():
    """
    GetCameraParameters() -> P, K, R, t, distCoeffs

    Returns all parameters of connected calibrated cameras.
    Returns: P: a 3x4 floating-point projection matrix.
             K: a 3x3 floating-point camera matrix.
             R: vector of rotation vectors estimated for each pattern view.
             t: vector of translation vectors estimated for each pattern view.
             distCoeffs: vector of distortion coefficients of 4, 5, or 8 elements.
    Parameters: This method does not have any parameter.

    Usage: parameters = SIGBTools.GetCameraParameters()
           P, K, R, t, distCoeffs = SIGBTools.GetCameraParameters()
    """
    P = CaptureManager.Instance.Parameters.P
    K = CaptureManager.Instance.Parameters.K
    R = CaptureManager.Instance.Parameters.R
    t = CaptureManager.Instance.Parameters.t
    distCoeffs = CaptureManager.Instance.Parameters.DistCoeffs

    return P, K, R, t, distCoeffs

def SetCameraParameters(P, K, R, t, distCoeffs):
    """
    SetCameraParameters(P, K, R, t, distCoeffs)

    Returns all parameters of connected calibrated cameras.
    Returns: This method does not return anything.
    Parameters: P: a 3x4 floating-point projection matrix.
                K: a 3x3 floating-point camera matrix.
                R: vector of rotation vectors estimated for each pattern view.
                t: vector of translation vectors estimated for each pattern view.
                distCoeffs: vector of distortion coefficients of 4, 5, or 8 elements.

    Usage: SIGBTools.SetCameraParameters(parameters)
           SIGBTools.SetCameraParameters(P, K, R, t, distCoeffs)
    """
    CaptureManager.Instance.Parameters.P = P
    CaptureManager.Instance.Parameters.K = K
    CaptureManager.Instance.Parameters.R = R
    CaptureManager.Instance.Parameters.t = t
    CaptureManager.Instance.Parameters.DistCoeffs = distCoeffs

#----------------------------------------------------------------------#
#                     VideoCaptureDevices Methods                      #
#----------------------------------------------------------------------#
def VideoCapture(idn=0, enum=CAMERA_VIDEOCAPTURE_640X480_30FPS):
    """
    VideoCapture(idn=0, enum=CAMERA_VIDEOCAPTURE_640X480_30FPS) -> True or False

    Class for video capturing from video files, image sequences or cameras. The class provides access to OpenCV for capturing multiple videos from cameras or for reading video files and image sequences.
    Returns: A boolean value inform if the video capture device was correct connected.
    Parameters: idn: camera index or full path of a image/video file,
                enum: enumeration for a connected camera [Format: SIGBTools.CAMERA_X_Y_Z, where X is the camera model, Y is the camera resolution and Z the number of captured frames per second].

    Usage: SIGBTools.VideoCapture()
           SIGBTools.VideoCapture(0)
           SIGBTools.VideoCapture(1, SIGBTools.CAMERA_PS3EYE_640X480_75FPS)
    """
    return CaptureManager.Instance.AddCamera(idn, enum)

def read():
    """
    read() -> images

    Grabs, decodes and returns the next video frame.
    Returns: A vector of synchronized images.
    Parameters: This method does not have any parameter.

    Usage: images = SIGBTools.read()
    """
    return CaptureManager.Instance.Read()

def release():
    """
    release()

    Closes video file or capturing device.
    Returns: This method does not return anything.
    Parameters: This method does not have any parameter.

    Usage: SIGBTools.release()
    """
    CaptureManager.Instance.Release()

def videoFPS():
    """
    videoFPS() -> fps

    Estimate the current number of captured frames per second.
    Returns: An integer value inform the current framerate.
    Parameters: This method does not have any parameter.

    Usage: fps = SIGBTools.videoFPS()
    """
    return CaptureManager.Instance.FPS

def videoSize():
    """
    videoSize() -> (width, height)

    Get the camera resolution.
    Returns: A map with two integer values, i.e. width and height.
    Parameters: This method does not have any parameter.

    Usage: width, heigh = SIGBTools.videoSize()
           size = SIGBTools.videoSize()
    """
    return CaptureManager.Instance.Size
