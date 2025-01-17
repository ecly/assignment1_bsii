#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : EyeFeatureDetector.py                                    -->
#<!-- Description: Class used for detecting some eye feature (e.g. pupil,   -->
#<!--            : glints, iris, among others) from the input videos        -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 13/02/2017                                               -->
#<!-- Change     : 13/02/2017 - Creation of these classes                   -->
#<!-- Review     : 13/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021301 $"

########################################################################
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance

from GeometricMethods import *
from RegionProps import RegionProps

########################################################################
class EyeFeatureDetector(object):
    """
    Class used for detecting some eye feature (e.g. pupil, glints, iris, among
    others) from the input videos.
    """

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Props(self):
        """Get a descriptors of contour-based connected components."""
        return self.__Props

    #----------------------------------------------------------------------#
    #                 EyeFeatureDetector Class Constructor                 #
    #----------------------------------------------------------------------#
    def __init__(self):
        """EyeFeatureDetector Class Constructor."""
        self.__Props = RegionProps()
        self.__firstPlot = True

    def __del__(self):
        """EyeFeatureDetector Class Destructor."""
        pass

    #----------------------------------------------------------------------#
    #                   Incomplete Public Class Methods                    #
    #----------------------------------------------------------------------#
    def getPupil(self, image, threshold=0, pupilMinimum=10, pupilMaximum=50):
        """
        Given an image, return the coordinates of the pupil candidates.
        """
        # Create the output variable.
        bestPupil = -1
        ellipses  = []
        centers   = []

        # Create variables to plot the regression data.
        # TIPS: You must select two blob properties and add their values in
        #       the following lists. Then, call the private method
        #       __plotRegressionData() in the end of your implementation.
        x = []
        y = []

        # Grayscale image.
        grayscale = image.copy()
        if len(grayscale.shape) == 3:
            grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

        # Define the minimum and maximum size of the detected blob.
        pupilMinimum = int(round(math.pi * math.pow(pupilMinimum, 2)))
        pupilMaximum = int(round(math.pi * math.pow(pupilMaximum, 2)))

        # Create a binary image.
        _, thres = cv2.threshold(grayscale, threshold, 255,
                                 cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((20,20), np.uint8)
        thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
        
        # Find blobs in the input image.
        _, contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->
        props = RegionProps()
        largestArea = 0
        areas = list()
        extends = list()
        for contour in contours:
            prop = props.calcContourProperties(contour, ["Centroid", "Area", "Extend"])
            ellipse = cv2.fitEllipse(contour) if (len(contour) > 4) else cv2.minAreaRect(contour)
            
            centroid = prop.get("Centroid")
            area = prop.get("Area")
            extend = prop.get("Extend")
            if(area > pupilMinimum and area < pupilMaximum and extend > 0.5):
                ellipses.append(ellipse)
                centers.append(centroid)
                extends.append(extend)
                areas.append(area)
                
                if (area > largestArea):#define best pupil as largest area
                    largestArea = area
                    bestPupil = len(ellipses) - 1
                    
        self.__plotData(extends, areas, bestPupil)
                    
        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

        # Return the final result.
        return ellipses, centers, bestPupil

    def getGlints(self, image, threshold=0, glintsMinimum=1, glintsMaximum=10,
                  numOfGlints=1, pupilCenter=(0, 0)):
        """
        Given an image, return the coordinates of the glints candidates.
        """
        # Correct the number of glints.
        if numOfGlints < 1:
            numOfGlints = 1
            
        # Define the minimum and maximum size of the detected blob.
        glintsMinimum = int(round(math.pi * math.pow(glintsMinimum, 2)))
        glintsMaximum = int(round(math.pi * math.pow(glintsMaximum, 2)))

        # Create the output variable.
        bestGlints= [-1] * numOfGlints
        ellipses  = []
        centers   = []

        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->

        # Grayscale image.
        grayscale = image.copy()
        if len(grayscale.shape) == 3:
            grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

        # Create a binary image.
        _, thres = cv2.threshold(grayscale, threshold, 255,
                                 cv2.THRESH_BINARY)#using binary instead of inv
        
        kernel = np.ones((5,5), np.uint8)
        thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
        
        # Find blobs in the input image.
        _, contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        
        props = RegionProps()
        for contour in contours:
            prop = props.calcContourProperties(contour, ["Centroid", "Area", "Extend", "BoundingBox"])
            ellipse = cv2.fitEllipse(contour) if (len(contour) > 4) else cv2.minAreaRect(contour)
            centroid = prop.get("Centroid")
            area = prop.get("Area")
           
            if (area > glintsMinimum):#basically disregards noise
                ellipses.append(ellipse)                    
                centers.append(centroid)                    
                
                
        #sort on distance to pupil
        ellipses.sort(key=lambda ellipse: self.__Distance(ellipse[0], pupilCenter)) 
        centers = map(lambda ellipse: (ellipse[0][0], ellipse[0][1]), ellipses)
        numOfGlints = numOfGlints if len(ellipses) >= numOfGlints else len(ellipses)#no more best than found
        bestGlints = [i for i in range (numOfGlints)]#since the returned array is sorted
        
        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

        # Return the final result.
        return ellipses, centers, bestGlints

    def getIris(self, image, pupilCenter, pupilRadius, numOfPoints=30):
        """
        Given an image, return the coordinates of the iris candidates.
        """
        # Create the output variable.
        lines = []
        points = []
        ellipse = None

        # Get the gradient info from the input image.
        gradient, orientation, magnitude = self.__GetGradientInfo(image)

        # Get the points distribuition around one circle.
        circle = getCircleSamples(pupilCenter, pupilRadius * 2, numOfPoints)
    
        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->
        maxVals = []
        for vals in circle:
            point = (vals[0][0], vals[0][1])
            direction = (point[0] - pupilCenter[0],point[1] - pupilCenter[1])
            length = math.sqrt(math.pow(direction[0], 2) + math.pow(direction[1], 2))
            direction = (direction[0] / length, direction[1] / length)#normalised
            scaledRadius = 1.3 * pupilRadius;# to ensure point is moved away from pupil entirely
            movedPupilCenter = (pupilCenter[0] + direction[0] * scaledRadius, pupilCenter[1] + direction[1] * scaledRadius)
            
            max = self.__FindMaxGradientValueOnNormal(magnitude, orientation, movedPupilCenter, (vals[0][0], vals[0][1]))
            maxVals.append(max)
            
        points = np.asarray(maxVals)
        
        ellipse = cv2.fitEllipse(points)
        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

        return ellipse, lines, points

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def getProcessedImage(self, image, threshold, pupilsEllipses=None,
                          pupilsCenters=None, bestPupil=None,
                          glintsEllipses=None, glintsCenters=None,
                          bestGlints=None, irisEllipse=None,
                          irisLines=None, irisPoints=None):
        """"
        Given an image and some eye feature coordinates, return the processed
        image.        """
        # Create the output variable
        processed = np.zeros(image.shape, np.uint8)

        # Baseline image and image resolution.
        result = image.copy()
        height, width = image.shape[:2]

        # Copy the original image for the first position.
        processed[:height/2, :width/2, :] = cv2.resize(result,
                                                       (width/2, height/2))

        # Copy the threshold image for the second position.
        grayscale = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if glintsEllipses == None and irisEllipse == None:
            _, thres = cv2.threshold(grayscale, threshold, 255,
                                 cv2.THRESH_BINARY_INV)
        elif irisEllipse == None:
            _, thres = cv2.threshold(grayscale, threshold, 255,
                                 cv2.THRESH_BINARY)
        else:
            thres = self.__GetSobel(grayscale)

        thres = cv2.cvtColor(thres, cv2.COLOR_GRAY2BGR)
        processed[:height/2, width/2:, :] = cv2.resize(thres,
                                                       (width/2, height/2))

        # Copy all candidates to the third position.
        candidates = result.copy()

        if (pupilsEllipses != None and len(pupilsEllipses) > 0 and
            pupilsCenters  != None and len(pupilsCenters) > 0):
            for pupil, center in zip(pupilsEllipses, pupilsCenters):
                self.__DrawCenter(candidates, center, (0, 0, 255))
                self.__DrawEllipse(candidates, pupil, (0, 0, 255))

        if (glintsEllipses != None and len(glintsEllipses) > 0 and
            glintsCenters  != None and len(glintsCenters) > 0):
            for glint, center in zip(glintsEllipses, glintsCenters):
                self.__DrawCenter(candidates, center, (0, 255, 0))
                self.__DrawEllipse(candidates, glint, (0, 255, 0))
    
        elif (irisLines != None and irisPoints != None):
            for line, point in zip(irisLines, irisPoints):
                cv2.line(candidates, line[0], line[1], (0, 255, 0), 2)
                cv2.circle(candidates, point, 2, (255, 0, 0), 5)

        processed[height/2:, :width/2, :] = cv2.resize(candidates,
                                                       (width/2, height/2))

        # Copy the best pupil candidate to the fourth position.
        if bestPupil != -1:
            self.__DrawCenter(result, pupilsCenters[bestPupil], (0, 0, 255))
            self.__DrawEllipse(result, pupilsEllipses[bestPupil], (0, 0, 255))

        if bestGlints != None:
            for best in bestGlints:
                if best != -1:
                    self.__DrawCenter(result, glintsCenters[best], (0, 255, 0))
                    self.__DrawEllipse(result, glintsEllipses[best], (0, 255, 0))

        elif irisEllipse != None:
            self.__DrawEllipse(result, irisEllipse, (0, 255, 0))

        processed[height/2:, width/2:, :] = cv2.resize(result,
                                                       (width/2, height/2))

        # Return the final result.
        return processed

    #----------------------------------------------------------------------#
    #                   Incomplete Private Class Methods                   #
    #----------------------------------------------------------------------#
    def __Distance(self, p1, p2):
        """Get the Euclidean distance between."""
        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->
        
        return scipy.spatial.distance.euclidean(p1, p2)

        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

    def __FindMaxGradientValueOnNormal(self, magnitude, orientation, p1, p2):
        """
        Find the boundary of the iris using the maximum gradient magnitude
        along the normals.
        """
        # Create the output variable.
        maxPoint = (0, 0)

        # Get integer coordinates on the straight line between p1 and p2.
        points = getLineCoordinates(p1, p2)
        try:
            normalVals = magnitude[points[:, 1], points[:, 0]]
        except:
            return maxPoint
    
        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->
        curveNormal = (p2[0]-p1[0], p2[1]-p1[1])#direction from center of pupil to periphery point
        curveNormalLen = math.sqrt(math.pow(curveNormal[0], 2) + math.pow(curveNormal[1], 2))
        curveNormalNormalised = (curveNormal[0] / curveNormalLen, curveNormal[1] / curveNormalLen)
        curveDirectionInDegrees = math.atan2(curveNormalNormalised[0], curveNormalNormalised[1]) * 180 / math.pi
        
        maxVal = -1
        index = -1
        for i,val in enumerate(normalVals):
            if val >= maxVal:
                point = points[i]
                if (point[0] < len(orientation) and point[1] < len(orientation[0])):
                    currentOrientation = orientation[point[0]][point[1]]
                    if (abs(currentOrientation - curveDirectionInDegrees) < 50):
                        index = i
                        maxVal = val
            
        maxPoint = points[index]
        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

        # Return the final result.
        return maxPoint

    def __GetGradientInfo(self, image):
        """
        Get the image gradient, the gradient magnitude and the gradient
        directions from an input image.
        """
        # Create the output variable.
        gradient = [np.zeros(image.shape[:2])] * 2
        orientation = np.zeros(image.shape[:2])
        magnitude = np.zeros(image.shape[:2])

        # Grayscale image.
        grayscale = image.copy()
        if len(grayscale.shape) == 3:
            grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=5)
        
        gradient = np.gradient(grayscale)
        magnitude = cv2.magnitude(sobelx, sobely)
        orientation = cv2.phase(sobelx, sobely, angleInDegrees=True)

        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

        # Return the final result.
        return gradient, orientation, magnitude

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __DrawCenter(self, image, center, color):
        """Drawing a cross over a determined region in the processed image."""
        if center[0] != -1 and center[1] != -1:
            cv2.circle(image, (int(center[0]), int(center[1])), 5, color, -1)

    def __DrawEllipse(self, image, rectangule, color):
        """Drawing an ellipse around a detected blob."""
        cv2.ellipse(image, rectangule, color, 2)
        points = cv2.boxPoints(rectangule)
        for i in range(4):
            cv2.line(image, tuple(np.array(points[i], np.int32)),
                     tuple(np.array(points[(i + 1) % 4], np.int32)), color, 2)

    def __GetSobel(self, image):
        """Get the Sobel borders using OpenCV."""
        # Gradient X.
        gradX = cv2.Sobel(image, cv2.CV_64F, 1, 0)

        # Gradient Y.
        gradY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

        # Converting back to uint8 dtype.
        absX = cv2.convertScaleAbs(gradX)
        absY = cv2.convertScaleAbs(gradY)

        # Merge the horizontal and vertical components.
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    def __plotData(self, x, y, bestPupil):
        """Plot the regression data based on blob properties."""
        # Check if there are valid regression data.
        if bestPupil == -1:
            return

        # Check if the user will plot for the first time.
        if self.__firstPlot:
            self.__fig, self.__ax = plt.subplots()
            self.__line, = self.__ax.plot([0, x[bestPupil]], [0, y[bestPupil]], "ko")
            plt.show(block=False)
            self.__firstPlot = False

        # Plot the regression data.
        self.__line.set_xdata(x)
        self.__line.set_ydata(y)
        self.__ax.plot(x[bestPupil], y[bestPupil], "ro")
        self.__ax.draw_artist(self.__ax.patch)
        self.__ax.draw_artist(self.__line)
        self.__fig.canvas.blit(self.__ax.bbox)
        self.__fig.canvas.flush_events()
