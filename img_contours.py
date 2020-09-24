from cv2 import findContours, contourArea, minEnclosingCircle, moments, RETR_LIST, RETR_TREE, RETR_CCOMP, RETR_EXTERNAL, \
    CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, convexHull, convexityDefects, fillConvexPoly
from numpy import ndarray

from img_draw import drawPoint, drawLine
from math_util import distanceBetweenPoints

CONTOURS_HIEARARCHY = [RETR_LIST, RETR_TREE, RETR_CCOMP, RETR_EXTERNAL]
CONTOURS_APPROX = [CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE]


def findImageContours(image: ndarray, mode: int = RETR_EXTERNAL, method: int = CHAIN_APPROX_NONE) -> list:
    """
    Return list of found contours in the input image.
    :param image: input image
    :param mode: contour hierarchy
    :param method: contour approximation method - SIMPLE or NONE
    :return:
    """
    assert image is not None and isinstance(image, ndarray)
    assert mode in CONTOURS_HIEARARCHY, "Invalid contour hierarchy type."
    assert method in CONTOURS_APPROX, "Invalid contour approximation method."
    return findContours(image, mode, method)[0]


def findMaxContour(contours: list) -> ndarray:
    """
    Returns contour with the largest area of all contours.
    :param contours: list of contours
    :return: contour with the max area
    """
    return max(contours, key=lambda c: contourArea(c))


def findMinEnclosingCircle(contour: ndarray) -> tuple:
    """
    Find minimum enclosing circle of the input contour.
    :param contour: contour that should be enclosed
    :return: center and radius of the minimum circle that encloses contour
    """
    (x, y), radius = minEnclosingCircle(contour)
    # TODO: validate - if empty contour is given
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def findContourExtremePoints(contour: ndarray) -> tuple:
    """
    Find extreme points in the given contour.
    :param contour: input contour
    :return: extreme points (left, right, top, bottom)
    """
    left = tuple(contour[contour[:, :, 0].argmin()][0])
    right = tuple(contour[contour[:, :, 0].argmax()][0])
    top = tuple(contour[contour[:, :, 1].argmin()][0])
    bottom = tuple(contour[contour[:, :, 1].argmax()][0])
    return left, right, top, bottom


def findContourCentre(contour: ndarray) -> tuple:
    """
    Find centre of the contour - using contour moments
    :param contour: contour which centre we're looking for
    :return: coordinates of the contour centre point
    """
    M = moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


# hull and contour defects
def calculateConsecutivePointsDist(points):
    """
    Calculates distances between consecutive points
    :param points: list of points that make contour defects
    :return: list of distances between consecutive points
    """
    consecutiveDist = []
    prev = points[0]
    begin = points[0]
    for i in range(1, len(points)):
        start = points[i]
        consecutiveDist.append(distanceBetweenPoints(prev, start))
        prev = start
    consecutiveDist.append(distanceBetweenPoints(prev, begin))
    return consecutiveDist


def getConvexHull(contour):
    return convexHull(contour, returnPoints=False)


def findConvexityDefects(contour):
    hull = getConvexHull(contour)
    return convexityDefects(contour, hull)


def getConvexPoints(contour, defects, type="start"):
    points = []
    for defect in defects:
        s, e, f, d = defect[0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        if type == "start":
            point = start
        elif type == "end":
            point = end
        elif type == "far":
            point = far
        else:
            point = (start, end, far)
        points.append(point)
    return points


def drawConvexHull(image, points, pointsColor=(255, 128, 64), lineColor=(0, 255, 0)):
    for point in points:
        drawPoint(image, point[0], 5, color=pointsColor)
        drawLine(image, point[0], point[1], color=lineColor)


def fillConvex(image, contour, color):
    fillConvexPoly(image, contour, color)
