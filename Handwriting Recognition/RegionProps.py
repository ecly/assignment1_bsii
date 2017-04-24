#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : RegionProps.py                                           -->
#<!-- Description: Getting descriptors of contour-based connected components-->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 25/02/2016                                               -->
#<!-- Change     : 25/02/2016 - Creation of this class                      -->
#<!-- Review     : 13/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021301 $"

########################################################################
import cv2
import math

########################################################################
class RegionProps(object):
    """
    This class is used for getting descriptors of contour-based connected
    components.

    The main method to use is: calcContourProperties(contour, properties=[]):
    contour: the contours variable found by cv2.findContours()
    properties: list of strings specifying which properties should be
                calculated and returned

    The following properties can be specified:
    Area: Area within the contour - float
    Boundingbox: Bounding box around contour - 4 tuple (topleft.x, topleft.y,
                                                        width, height)
    Length: Length of the contour
    Centroid: The center of contour - (x, y)
    Moments: Dictionary of moments
    Perimiter: Permiter of the contour - equivalent to the length
    Extend: Ratio of the area and the area of the bounding box. Expresses how
            spread out the contour is
    Convexhull: Calculates the convex hull of the contour points
    IsConvex: boolean value specifying if the set of contour points is convex

    Returns: Dictionary with key equal to the property name

    Example: 
        image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        goodContours = []
        for cnt in contours:
            vals = props.calcContourProperties(cnt, ["Area", "Length",
                                                     "Centroid", "Extend",
                                                    "ConvexHull"])
            if vals["Area"] > 100 and vals["Area"] < 200:
                goodContours.append(cnt)
    """

    #----------------------------------------------------------------------#
    #                    RegionProps Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """RegionProps Class Constructor."""
        pass

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def calcContourProperties(self, contour, properties=[]):
        """Calcule and return a list of strings specifying by properties."""
        # Initial variables.
        failInInput  = False
        props = {}

        for prop in properties:
            prop = str(prop).lower()
            moments = cv2.moments(contour)
            if prop == "area":
                props.update({"Area" : self.__CalcArea(contour)})
            elif prop == "boundingbox":
                props.update({"BoundingBox" : self.__CalcBoundingBox(contour)})
            elif prop == "length":
                props.update({"Length" : self.__CalcLength(contour)})
            elif prop == "centroid":
                props.update({"Centroid" : self.__CalcCentroid(moments)})
            elif prop == "moments":
                props.update({"Moments" : moments})
            elif prop == "perimiter":
                props.update({"Perimiter" : self.__CalcPerimiter(contour)})
            elif prop == "extend":
                props.update({"Extend" : self.__CalcExtend(contour)})
            elif prop == "convexhull":
                props.update({"ConvexHull" : self.__CalcConvexHull(contour)})
            elif prop == "isconvex":
                props.update({"IsConvex" : self.__IsConvex(contour)})
            elif failInInput:
                pass
            else:
                print "\t--" * 20
                print "\t*** PROPERTY ERROR " + prop + " DOES NOT EXIST ***"
                print "\tTHIS ERROR MESSAGE WILL ONLY BE PRINTED ONCE"
                print "\--" * 20
                failInInput = True

        return props

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __CalcArea(self, contour):
        """Calculates a contour area."""
        return cv2.contourArea(contour)

    def __CalcBoundingBox(self, points):
        """Calculates the up-right bounding rectangle of a point set."""
        return cv2.boundingRect(points)

    def __CalcLength(self, curve):
        """Calculates a contour perimeter or a curve length."""
        return cv2.arcLength(curve, True)

    def __CalcCentroid(self, moments):
        """
        Calculates the centroid of the moments up to the third order of a
        polygon or rasterized shape.
        """
        retVal = (-1, -1)        
        if moments["m00"] != 0:
            retVal =  (moments["m10"] / moments["m00"],
                       moments["m01"] / moments["m00"])
        return retVal

    def __CalcPerimiter(self, curve):
        """Calculates a contour perimeter or a curve length."""
        return cv2.arcLength(curve, True)

    def __CalcExtend(self, contour):
        area = self.__CalcArea(contour)
        boundingBox = self.__CalcBoundingBox(contour)
        return area / (boundingBox[2] * boundingBox[3])

    def __CalcConvexHull(self, points):
        """Finds the convex hull of a point set."""
        return cv2.convexHull(points)

    def __IsConvex(self, contour):
        """Tests a contour convexity."""
        return cv2.isContourConvex(contour)
