#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : FrameRate.py                                             -->
#<!-- Description: Class used for calculating the number of frames analyzed -->
#<!--            : per second (FPS) by the Python application               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 23/06/2014                                               -->
#<!-- Change     : 23/06/2014 - Creation of these classes                   -->
#<!--            : 10/02/2017 - Convert the class for IAML course           -->
#<!-- Review     : 10/02/2017 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2017021001 $"

########################################################################
import time

########################################################################
class FrameRate(object):
    """FrameRate Class is used for calculating the frames per second (FPS)."""

    #----------------------------------------------------------------------#
    #                     FrameRate Class Constructor                      #
    #----------------------------------------------------------------------#
    def __init__(self):
        """FrameRate Class Constructor"""
        self.__timestamp = int(round(time.time() * 1000))
        self.__current   = 0
        self.__counter   = 0

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def calculateFPS(self):
        """Calculate the number of frames per second of the process."""
        timestamp = int(round(time.time() * 1000)) - self.__timestamp

        if timestamp < 1000:
            self.__counter += 1
        else:
            self.__current = self.__counter
            self.__timestamp = int(round(time.time() * 1000))
            self.__counter = 0

        return self.__current
