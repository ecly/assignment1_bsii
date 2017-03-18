#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : SkinColorDetector.py                                     -->
#<!-- Description: Class used for detecting automatically the skin color    -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 27/02/2017                                               -->
#<!-- Change     : 27/02/2017 - Creation of these classes                   -->
#<!-- Review     : 01/03/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017030101 $"

########################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np

########################################################################
class SkinColorDetector(object):
    """
    Class used for detecting automatically the best threshold for a skin color
    detection approach.
    """

    #----------------------------------------------------------------------#
    #                 SkinColorDetector Class Constructor                  #
    #----------------------------------------------------------------------#
    def __init__(self, cascade):
        """SkinColorDetector Class Constructor."""
        try:
            self.__cascade = cv2.CascadeClassifier(cascade)
        except:
            exit(0)

    def __del__(self):
        """SkinColorDetector Class Destructor."""
        pass

    #----------------------------------------------------------------------#
    #                   Incomplete Public Class Methods                    #
    #----------------------------------------------------------------------#
    def getSkinColor(self, image):
        """Given an image, return with the detected human skin."""
        # Create the output variable.
        lower_value = 0
        upper_value = 0
        threshold   = 0

        # Create the final result image.
        result = image.copy()
        height, width = image.shape[:2]
    
        # Create an array to insert the detected faces coorditates as a
        # rectangle ([x, y] the upper left coordinate and [w, h] the size).
        faces = np.array([[0, 0, image.shape[1], image.shape[0]]])

        # Create an histogram representation.
        hist = np.zeros(image.shape, np.uint8)

        # Create the skin color image.
        skin = np.zeros(image.shape, np.uint8)

        # Convert the input image to grayscale.
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #<!--------------------------------------------------------------------------->
        #<!--                            YOUR CODE HERE                             -->
        #<!--------------------------------------------------------------------------->

        

        #<!--------------------------------------------------------------------------->
        #<!--                                                                       -->
        #<!--------------------------------------------------------------------------->

        # Create the image with the steps of skin color detector.
        result = self.__getProcessedImage(image, faces, hist, skin)

        # Return the result processed image.
        return result, faces, threshold, lower_value, upper_value

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __getProcessedImage(self, image, faces, hist, skin):
        """Given an image, return the processed image."""
        # Create the output variable
        processed = np.zeros(image.shape, np.uint8)

        # Baseline image and image resolution.
        result = image.copy()
        height, width = image.shape[:2]

        # Copy the original image for the first position.
        processed[:height/2, :width/2, :] = cv2.resize(result,
                                                       (width/2, height/2))

        # Draw a rectangle in the detected faces.
        result = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

        processed[:height/2, width/2:, :] = cv2.resize(result,
                                                       (width/2, height/2))

        # Show the histogram.
        resized = cv2.resize(hist, (width/2, height/2))
        if processed[height/2:, :width/2, :].shape == resized.shape:
            processed[height/2:, :width/2, :] = resized
        else:
            processed[height/2+1:, :width/2, :] = resized

        # Show the skin image.
        resized = cv2.resize(skin, (width/2, height/2))
        if processed[height/2:, width/2:, :].shape == resized.shape:
            processed[height/2:, width/2:, :] = resized
        else:
            processed[height/2+1:, width/2:, :] = resized

        # Return the final result.
        return processed

    def __getHistogramImage(self, histogram, size):
        """Generate a histogram image using OpenCV."""
        # Define the histogram size.
        result = np.zeros((300, 256, 3))

        # Vector of colors R, G and B.
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        # Grayscale levels distribuition.
        bins = np.arange(256).reshape(256, 1)

        # Draw the histogram curve for each channel.
        for uid, color in enumerate(colors):
            hist_item = np.int32(np.around(histogram[:, uid]))
            cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
            points = np.int32(np.column_stack((bins, hist_item)))
            cv2.polylines(result, [points], False, color)

        # Flip the histogram image in the up/down direction.
        result = np.flipud(result)

        # Return the final image.
        return cv2.resize(result, size)
