from math import sqrt, acos, atan, pi


def euclideanDistance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculates euclidean distance between points with the given coordinates.
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: distance between two points.
    """
    # print(type(x1), type(y1), type(x2),type(y2))
    # assert isinstance(x1, (float, int)) and isinstance(y1, (float, int)) and isinstance(x2, (float, int)) and \
    #        isinstance(y2, (float, int)), "Invalid arguments values - arguments must be numbers (integer or float)."
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distanceBetweenPoints(firstPoint: tuple, secondPoint: tuple) -> float:
    """
    Calculates euclidean distance between points with the given coordinates.
    :param firstPoint:  first point
    :param secondPoint: second point
    :return: distance between two points.
    """
    x1, y1 = firstPoint
    x2, y2 = secondPoint
    return euclideanDistance(x1, y1, x2, y2)


def calculateAngle(a: float, b: float, c: float) -> float:
    """
    Calculate angle (in radians) between edges b and c of triangle by cosine theorem.
    :param a: triangle edge
    :param b: triangle edge
    :param c: triangle edge
    :return: angle between edges b and c
    """
    # assert isinstance(a, (float, int)) and isinstance(b, (float, int)) and isinstance(c, (float, int)) and \
    #        a > 0 and b > 0 and c > 0, "Invalid arguments values - arguments must be positive numbers."
    # TODO: add check if these value can form triangle ( a + b > c....etc)
    return acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem


# NOTE - used only in test
def calculateAngleBetweenPoints(firstPoint, secondPoint):
    # TODO: add validation
    x1, y1 = firstPoint
    x2, y2 = secondPoint
    try:
        k = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError as e:
        print("Zero division detected")
        return 90 if (y2 - y1) > 0 else -90
        # return float("inf")
    return atan(k) * 180 / pi


def calculateAngleBetweenPoints2(firstPoint, secondPoint):
    x1, y1 = firstPoint
    x2, y2 = secondPoint
    norm1 = sqrt(x1 * x1 + y1 * y1)
    norm2 = sqrt(x2 * x2 + y2 * y2)
    dotProduct = x1 * x2 + y1 * y2
    if norm1 == norm2 == 0:
        return 0
    if norm1 == 0:
        return 90 - (atan(y2 / x2) * 180 // pi)
    if norm2 == 0:
        return 90 - (atan(y1 / x1) * 180 // pi)
    a = dotProduct / (norm1 * norm2)
    if a >= 1.0:
        angle = 0.0
    elif a <= -1.0:
        angle = pi
    else:
        angle = acos(a)
    return 90 - angle * 180 // pi


def calculateAngleBetweenPoints3(firstPoint, secondPoint):
    # TODO: add validation
    if firstPoint[1] < secondPoint[1]:
        x1, y1 = firstPoint
        x2, y2 = secondPoint
    else:
        x1, y1 = secondPoint
        x2, y2 = firstPoint
    try:
        k = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError as e:
        print("Zero division detected")
        return 90 if (y2 - y1) > 0 else -90
        # return float("inf")
    return atan(k) * 180 / pi


def calculateLineSlope(firstPoint, secondPoint):
    c = distanceBetweenPoints(firstPoint, secondPoint)
    b = abs(firstPoint[0] - secondPoint[0])
    a = abs(firstPoint[1] - secondPoint[1])
    angle = calculateAngle(a, b, c)
    return angle * 180 // pi




