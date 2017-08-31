#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : GeometricMethods.py                                      -->
#<!-- Description: Methods to calculate some geometric transformations      -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 23/02/2017                                               -->
#<!-- Change     : 23/02/2017 - Creation of this class                      -->
#<!-- Review     : 23/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017022301 $"

########################################################################
import math
import numpy as np

########################################################################
def getCircleSamples(center=(0, 0), radius=1, numOfPoints=30):
    """
    Sample a circle with center center = (x, y), radius = 1 and N
    numOfPoints in the circle. It will return an array of a tuple
    containing the points (x, y) in the circle and the curve gradient in
    the point (d_x, d_y).
    """
    sample = np.linspace(0, 2 * math.pi, numOfPoints)
    return [((radius * np.cos(t) + center[0],
              radius * np.sin(t) + center[1]),
              np.cos(t), np.sin(t)) for t in sample]

def getLineCoordinates(p1, p2):
    """
    Get integer coordinates between p1 and p2 using Bresenhams algorithm.
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
