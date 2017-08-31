#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : ROISelector.py                                           -->
#<!-- Description: Select a Region of Interested (RoI) in the input image   -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 25/02/2016                                               -->
#<!-- Change     : 25/02/2016 - Creation of this class                      -->
#<!-- Review     : 25/02/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016022501 $"

########################################################################
import cv2

########################################################################
class ROISelector(object):
    """This class returns the corners of the selected area as: [(UpperLeftcorner), (LowerRightCorner)].
       Use the Right Mouse Button to set upper left hand corner and the Left Mouse Button to set the lower right corner.
        Click on the image to set the area
        Keys:
            Enter/SPACE - OK
            ESC/q       - Exit (Cancel)
    """

    #----------------------------------------------------------------------#
    #                    ROISelector Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self, image):
        """ROISelector Class Constructor."""
        self.__image = image.copy()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def SelectArea(self, winName="Select an area", winPos=(400, 400)):
        """This method returns the corners of the selected area as: [(UpLeftcorner), (DownRightCorner)]."""
        # Reset the selected points.
        self.__ResetPoints()

        # Define the window name.
        self.__winName = winName

        # Update and show the input image.
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(winName, self.__OnMouseOver)
        cv2.moveWindow(winName, winPos[0], winPos[1])
        self.__Update()

        # Main while condition
        while True:
            # Read the keyboard selection.
            ch = cv2.waitKey(1)

            if ch is 27 or ch is ord("q"):
                cv2.destroyWindow(winName)
                return None, False
            elif ch is 13 or ch is 32:
                corners = self.__SetCorners()
                if corners is None:
                    continue
                cv2.destroyWindow(winName)
                return corners, True

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __ResetPoints(self):
        """Reset the point from selected corners."""
        self.leftPoints  = None
        self.rightPoints = None

    def __Update(self):
        """Update the current processed image."""
        # Check if there are valid values.
        if self.leftPoints is None or self.rightPoints is None:
            cv2.imshow(self.__winName, self.__image)
            return

        # Draw a red rectangle in the selected area.
        image = self.__image.copy()
        cv2.rectangle(image, self.leftPoints, self.rightPoints, (0, 0, 255), 1)
        cv2.imshow(self.__winName, image)

    def __SetCorners(self):
        """Convert the selected corners in a valid vector."""
        # Check if there are valid values.
        if self.leftPoints is None or self.rightPoints is None:
            return None

        # Calculates the corners points.
        upLeft    = (min(self.leftPoints[0], self.rightPoints[0]), min(self.leftPoints[1], self.rightPoints[1]))
        downRight = (max(self.leftPoints[0], self.rightPoints[0]), max(self.leftPoints[1], self.rightPoints[1]))

        # Create and return the vector.
        points = []
        points.append(upLeft)
        points.append(downRight)
        return points

    #----------------------------------------------------------------------#
    #                        Windows Events Methods                        #
    #----------------------------------------------------------------------#
    def __OnMouseOver(self, event, x, y, flags, param):
        """Get all mouse events over the main windows."""
        if  flags & cv2.EVENT_FLAG_LBUTTON:
            self.leftPoints = x, y

        if  flags & cv2.EVENT_FLAG_RBUTTON:
            self.rightPoints = x, y

        self.__Update()
