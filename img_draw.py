from cv2 import circle, drawContours, LINE_AA, line, putText, FONT_HERSHEY_SIMPLEX
from numpy import ndarray


def drawCircle(image: ndarray, center: tuple, radius: float, color: tuple, thickness: int = 1) -> None:
    """
    Draws circle in the given image. (changes image)
    :param image: target image
    :param center: center of the circle
    :param radius: radius of the circle
    :param color: color of the circle
    :param thickness: line thickness
    :return: None, changes image in-place
    """
    # TODO: add type assertion
    circle(image, center, radius, color, thickness)


def drawImageContours(image: ndarray, contours: list, color: tuple = (255, 0, 0), thickness: int = 1,
                      lineType: int = LINE_AA, maxLevel: int = 1) -> None:
    """
    Draws contours outlines or filled contours.
    :param image: target image
    :param contours: list of contours that should be drawn
    :param color:  color of the contour line
    :param thickness: contour line thickness
    :param lineType: type of the line - OpenCV parameter, see: #LineTypes
    :param maxLevel: Maximal level for drawn contours. If it is 0, only the specified contour is drawn.
    .   If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
    .   draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
    .   parameter is only taken into account when there is hierarchy available.
    """
    drawContours(image, contours, -1, color, thickness, lineType, maxLevel=maxLevel)


def drawPoint(image, point, radius=5, color=(0, 0, 0), thickness=-1):
    circle(image, point, radius, color, thickness)


def drawLine(image, start, end, color=(0, 255, 0), thickness=2):
    line(image, start, end, color, thickness=thickness)


def drawPoints(image, points, radius=5, color=(0, 0, 0), thickness=-1):
    for point in points:
        drawPoint(image, point, radius, color, thickness)


def drawText(image, text):

    location = (50, 50)
    putText(
        image,
        text,
        location,
        FONT_HERSHEY_SIMPLEX,
        1,
        (80, 0, 209, 255),#color,#(209, 80, 0, 255),
        3)
