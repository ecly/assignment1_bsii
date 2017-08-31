#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : ImageManager.py                                          -->
#<!-- Description: Class used for managing the digital image processing     -->
#<!--              algorithms                                               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: This class is based on the Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 25/10/2015                                               -->
#<!-- Change     : 25/10/2015 - Creation of these classes                   -->
#<!-- Review     : 24/10/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015102501 $"

########################################################################
import cv2
import math
import numpy as np
import warnings

from pylab import close
from pylab import copy
from pylab import figure
from pylab import plot
from pylab import subplot
from pylab import title

from ClassProperty import ClassProperty

########################################################################
class ImageManager(object):
    """ImageManager Class is used for managing the digital image processing algorithms."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance for the image manager."""
        if self.__Instance is None:
            self.__Instance = Algorithms()
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
        return "Framework.ImageProcessing.ImageManager object."

########################################################################
class Algorithms(object):
    """Algorithms Class is used for managing the digital image processing algorithms."""

    #----------------------------------------------------------------------#
    #                     Algorithms Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Algorithms Class Constructor."""
        warnings.simplefilter("ignore")

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def GetHomographyFromMouse(self, image1, image2, N=4):
        """
        GetHomographyFromMouse(image1, image2, N=4) -> homography, mousePoints

        Calculates the homography from a plane in image "image1" to a plane in image "image2" by using the mouse to define corresponding points
        Returns: 3x3 homography matrix and a set of corresponding points used to define the homography
        Parameters: N >= 4 is the number of expected mouse points in each image,
                    when N < 0: then the corners of image "image1" will be used as input and thus only 4 mouse clicks are needed in image "image2".

        Usage: Use left click to select a point and right click to remove the most recently selected point.
        """
        # Vector with all input images.
        images = []
        images.append(cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB))
        images.append(cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2RGB))

        # Vector with the points selected in the input images.
        mousePoints = []
        # Control the number of processed images.
        firstImage = 0

        # When N < 0, then the corners of image "image1" will be used as input.
        if N < 0:
            # Force 4 points to be selected.
            N = 4
            firstImage = 1
            m, n = image1.shape[0:2]
            # Define corner points from image "image1".
            mousePoints.append([(0, 0), (n, 0), (n, m), (0, m)])

        # Check if there is the minimum number of needed points to estimate the homography.
        if math.fabs(N) < 4:
            N = 4
            print("At least 4 points are needed!!!")

        # Make a pylab figure window.
        fig = figure(1)

        # Get the correspoding points from the input images.
        for i in range(firstImage, 2):
            # Setup the pylab subplot.
            ax = subplot(1, 2, i+1)
            ax.imshow(images[i])
            ax.axis("image")
            title("Click " + str(N) + " times in this image.")
            fig.canvas.draw()
            ax.hold("On")

            # Get mouse inputs.
            mousePoints.append(fig.ginput(N, -1))

            # Draw selected points in the processed image.
            for point in mousePoints[i]:
                cv2.circle(images[i], (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
            ax.imshow(images[i])
            for (x, y) in mousePoints[i]:
                plot(x, y, "rx")
            fig.canvas.draw()

        # Close the pylab figure window.
        close(fig)

        # Convert to OpenCV format.
        points1 = np.array([[x, y] for (x, y) in mousePoints[0]])
        points2 = np.array([[x, y] for (x, y) in mousePoints[1]])

        # Calculate the homography.
        homography, mask = cv2.findHomography(points1, points2)
        return homography, mousePoints

    def FrameTrackingData2BoxData(self, data):
        """Convert a row of points into tuple of points for each rectangle."""
        points = [(int(data[i]), int(data[i+1])) for i in range(0, 11, 2)]
        boxes  = []
        for i in range(0, 6, 2):
            box = tuple(points[i:i+2])
            boxes.append(box)
        return boxes

    def GetHomographyTG(self, image, homography, texture, scale):
        pass

    def TextureMoving(self, image, gridPoints, texture):
        """Develop a way of solving the problem so that the texture is mapped even during rotations."""
        pass

    def DetectPlaneObject(self, image, minSize=1000):
        """A simple attempt to detect rectangular color regions in the image."""
        # Convert the input image from RGB to HSV.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype("uint8")
        s = hsv[:, :, 1].astype("uint8")
        v = hsv[:, :, 2].astype("uint8")

        # Get each RGB channel from the input image.
        r = image[:, :, 0].astype("uint8")
        g = image[:, :, 1].astype("uint8")
        b = image[:, :, 2].astype("uint8")

        # Use the red channel for detection.
        output = (255 * (r > 230)).astype("uint8")

        # Show the result image.
        result = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        cv2.imshow("ColorDetection", output)

        # Find and return the squares of the input image.
        squares = self.__FindSquares(output, minSize)
        return squares

    def EstimateHomography(self, points1, points2):
        """
        EstimateHomography(points1, points2) -> homography

        Calculates a homography from one plane to another using a set of 4 points from each plane.
        Returns: 3x3 homography matrix and a set of corresponding points used to define the homography
        Parameters: points1 is 4 points (np.array) on a plane
                    points2 is the corresponding 4 points (np.array) on an other plane
        """
        # Check if there is sufficient points.
        if len(points1) == 4 and len(points2) == 4:
            # Get x, y values.
            x1, y1 = points1[0]
            x2, y2 = points1[1]
            x3, y3 = points1[2]
            x4, y4 = points1[3]

            # Get x tilde and y tilde values.
            x_1, y_1 = points2[0]
            x_2, y_2 = points2[1]
            x_3, y_3 = points2[2]
            x_4, y_4 = points2[3]

            # Create matrix A.
            A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1 * x_1, y1 * x_1, x_1],
                           [0, 0, 0, -x1, -y1, -1, x1 * y_1, y1 * y_1, y_1],
                           [-x2, -y2, -1, 0, 0, 0, x2 * x_2, y2 * x_2, x_2],
                           [0, 0, 0, -x2, -y2, -1, x2 * y_2, y2 * y_2, y_2],
                           [-x3, -y3, -1, 0, 0, 0, x3 * x_3, y3 * x_3, x_3],
                           [0, 0, 0, -x3, -y3, -1, x3 * y_3, y3 * y_3, y_3],
                           [-x4, -y4, -1, 0, 0, 0, x4 * x_4, y4 * x_4, x_4],
                           [0, 0, 0, -x4, -y4, -1, x4 * y_4, y4 * y_4, y_4]])

            # Calculate SVD.
            U, D, V = np.linalg.svd(A)

            # Get last row of V returned from SVD.
            h = V [8]

            # Create homography matrix.
            homography = np.matrix([[h[0, 0], h[0, 1], h[0, 2]],
                                    [h[0, 3], h[0, 4], h[0, 5]],
                                    [h[0, 6], h[0, 7], h[0, 8]]])

            # Normalize homography.
            homography /= homography[2, 2]

            # Return the homography matrix.
            return homography

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __FindSquares(self, image, minSize=2000, maxAngle=1):
        """Locate a rectangle in the image of minimum area, minSize, and maximum angle, maxAngle, between sides."""
        # A vector with all detected squares.
        squares = []

        # Find the contours of a detected object.
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze each detected contours curve.
        for cnt in contours:
            # Calculates a contour perimeter or a curve length.
            perimeter = cv2.arcLength(cnt, True)
            # Approximates a polygonal curve(s) with the specified precision.
            cnt = cv2.approxPolyDP(cnt, 0.08 * perimeter, True)
            # Check if this curve is a valid square.
            if len(cnt) == 4 and cv2.contourArea(cnt) > minSize and cv2.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([self.__GetAngleCos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in xrange(4)])
                if max_cos < maxAngle:
                    squares.append(cnt)

        # Return the squares of the input image.
        return squares

    def __GetAngleCos(self, p0, p1, p2):
        """Returns the cosine angle."""
        d1, d2 = p0-p1, p2-p1
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.ImageProcessing.Algorithms object."
