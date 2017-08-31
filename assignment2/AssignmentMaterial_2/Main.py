#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Main.py                                                  -->
#<!-- Description: Main class of the SIGB framework                         -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/10/2015                                               -->
#<!-- Change     : 24/10/2015 - Creation of this class                      -->
#<!-- Review     : 24/10/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015102401 $"

########################################################################
import sys

from Menu import Menu

########################################################################

#----------------------------------------------------------------------#
#                             Main Methods                             #
#----------------------------------------------------------------------#
def main(argv):
    Menu().Start()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
