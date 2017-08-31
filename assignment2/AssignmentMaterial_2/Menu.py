#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Menu.py                                                  -->
#<!-- Description: Main menu of this project                                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/10/2015                                               -->
#<!-- Change     : 24/10/2015 - Creation of this class                      -->
#<!-- Review     : 24/02/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016022401 $"

########################################################################
import os

from Assignments._02.Assignment2 import Assignment2

########################################################################
class Menu(object):
    """Menu class is the main menu of this project."""

    #----------------------------------------------------------------------#
    #                        Menu Class Constructor                        #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Menu Class Constructor."""
        self.__assignment2 = Assignment2()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Start(self):
        """Start SIGB framework."""
        option = "-1"
        clear  = lambda: os.system("cls" if os.name == "nt" else "clear")
        while option != "0":
            clear()
            print "\n\t#################################################################"
            print "\t#                                                               #"
            print "\t#      Introduction to Image Analysis and Machine Learning      #"
            print "\t#                                                               #"
            print "\t#################################################################\n"
            print "\t[1] Assignment #02 part 03."
            print "\t[0] Exit.\n"
            option = raw_input("\tSelect an option: ")

            if option == "1":
                self.__assignment2.Start()
            elif option != "0":
                raw_input("\n\tINVALID OPTION!!!")
