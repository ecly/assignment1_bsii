#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : AugmentedManager.py                                      -->
#<!-- Description: Class used for managing the augmented reality processing -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: This class is based on the Lazy Initialization examples  -->
#<!--              illustrated in Wikipedia                                 -->
#<!-- Date       : 05/12/2015                                               -->
#<!-- Change     : 05/12/2015 - Creation of these classes                   -->
#<!-- Review     : 05/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015120501 $"

########################################################################
from ClassProperty import ClassProperty

from Cube    import Cube
from Pattern import Pattern

########################################################################
class AugmentedManager(object):
    """AugmentedManager Class is used for managing the augmented reality processing."""

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
            self.__Instance = Augmented()
        return self.__Instance

    #----------------------------------------------------------------------#
    #                  AugmentedManager Class Constructor                  #
    #----------------------------------------------------------------------#
    def __init__(self):
        """This constructor is never used by the system."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Augmented.AugmentedManager object."

########################################################################
class Augmented(object):
    """Augmented Class is used for managing the augmented reality processing objects."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Cube(self):
        """Get the augmented cube object."""
        return self.__Cube

    @property
    def Pattern(self):
        """Get the pattern object."""
        return self.__Pattern

    #----------------------------------------------------------------------#
    #                      Augmented Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Augmented Class Constructor."""
        self.__Cube = Cube()
        self.__Pattern = Pattern()

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Augmented.Augmented object."
