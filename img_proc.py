import cv2
import numpy as np

from exceptions import InvalidArgumentTypeOrValueError

USUAL_COLORSPACES_MAP = {"grayscale": cv2.COLOR_BGR2GRAY, "hsv": cv2.COLOR_BGR2HSV}
THRESHOLD_TYPES = (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV,
                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def getColorSpaces():
    """
    :return: list of available color spaces in cv2
    :rtype: list
    """
    colSpaces = [i for i in dir(cv2) if i.startswith('COLOR_')]
    return colSpaces


def convertColorSpace(image: object, colorSpace: str = "grayscale") -> object:
    """
    Convert the input image to the given color space
    :param image: image that should be converted
    :type image: numpy array
    :param colorSpace:  target color space
    :type colorSpace: string
    :return: image with the changed colorspace
    :rtype: numpy array
    """
    if not isinstance(colorSpace, str):
        raise InvalidArgumentTypeOrValueError("Invalid color space argument type")
    colorSpace = USUAL_COLORSPACES_MAP.get(colorSpace, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, colorSpace)


def getImageUpperPart(image: object, ratio: int) -> object:
    """
     Returns only upper half of the image
    :param image: input image
    :param ratio: ratio of the image that should be cropped - upper part of the image
    :return: extracted part of the image
    """
    # if not isinstance(ratio, int) or ratio < 1:
    #     raise InvalidArgumentTypeOrValueError("Ratio of the image must be positive integer.")
    return image[0:int(image.shape[0] // ratio)]


def applyCannyEdge(image: object, thresholdLower: int, thresholdUpper: int, **otherParams: dict) -> object:
    """
    Applies Canny edge detection algorithm.
    :param image: input image that should be processed
    :type image: numpy object
    :param thresholdLower: lower threshold in Canny edge algorithm
    :type thresholdLower: int
    :param thresholdUpper: upper threshold in Canny edge algorithm
    :type: int
    :param otherParams: other optional params
    :type otherParams: dict
    :return: return image with detected edges
    :rtype: image object (numpy array)
    """
    if not (isinstance(thresholdLower, int) and isinstance(thresholdUpper, int)):
        raise InvalidArgumentTypeOrValueError("Threshold values must be integer type.")
    if not thresholdLower < thresholdUpper:
        raise InvalidArgumentTypeOrValueError("Lower threshold must be less than upper threshold.")
    return cv2.Canny(image, thresholdLower, thresholdUpper, **otherParams)


def applyDilation(image: np.ndarray, kernel: tuple = (3, 3), iterations: int = 1) -> np.ndarray:
    """
    Dilates the given image.
    :param image: image that will be dilated.
    :type image: image objec
    :param kernel: vector of kernel dimensions, preferred values in vector should be odd.
    :type kernel: vector types (tuple, list)
    :param iterations: number of applied dilations, at least 1 
    :type iterations: int
    :return: dilated image
    :rtype: image
    """
    assert isinstance(iterations,
                      int) and iterations > 0, "Invalid parameters format - number of iterations must be positive " \
                                               "integer "
    if kernel is not None and not (type(kernel) in (tuple, list) and len(kernel) == 2):
        raise InvalidArgumentTypeOrValueError("Kernel must be of vector type with 1x2 size")
    return cv2.dilate(image, kernel=kernel, iterations=iterations)


def applyErosion(image: np.ndarray, kernel: tuple = (3, 3), iterations: int = 1) -> np.ndarray:
    """
    Erodes the given image.
    :param image: image that will be eroded.
    :type image: image objec
    :param kernel: vector of kernel dimensions, preferred values in vector should be odd.
    :type kernel: vector types (tuple, list)
    :param iterations: number of applied dilations, at least 1 
    :type iterations: int
    :return: eroded image
    :rtype: image
    """
    assert isinstance(iterations,
                      int) and iterations > 0, "Invalid parameters format - number of iterations must be positive " \
                                               "integer "
    if kernel is not None and not (type(kernel) in (tuple, list) and len(kernel) == 2):
        raise InvalidArgumentTypeOrValueError("Kernel must be of vector type with 1x2 size")
    return cv2.erode(image, kernel=kernel, iterations=iterations)


def applyGaussianBlur(image: np.ndarray, kernel: tuple = (3, 3), sigmaX: float = 0, sigmaY: float = 0) -> object:
    """
    Applies Gaussian Blur on the image.
    :param image: image that will be blurred.
    :type image: image object
    :param kernel: vector of kernel dimensions, values in vector should be odd.
    :type kernel: vector types (tuple, list)
    :param sigmaX: stddev in X direction
    :type sigmaX: int
    :param sigmaY: stddev in Y direction
    :type sigmaY: stddev in Y direction
    :return: blurred image
    :rtype: image
    """
    # assert sigmaX is None or isinstance(sigmaX,
    #                                     float), "Invalid parameters format - sigmaX has not valid value or format"
    # assert sigmaY is None or isinstance(sigmaY,
    #                                     float), "Invalid parameters format - sigmaY has not valid value or format"

    assert type(kernel) in (tuple, list) and len(kernel) == 2 and kernel[0] % 2 == 1 and kernel[1] % 2 == 1, \
        "Kernel must be of vector type with 1x2 size with odd values elements."
    return cv2.GaussianBlur(image, kernel, sigmaX=sigmaX, sigmaY=sigmaY)


def binarizeImage(image, threshold=127, newValue=255, thresholdType=cv2.THRESH_BINARY):
    """
    Applies threshold, so it returns binary image.
    :param image: image that will be binary
    :param threshold: threshold value
    :param newValue: value to be set
    :param thresholdType: cv2 threshold type that will be used
    :rtype: image object
    :return: binary image
    """
    assert image is not None and isinstance(image, np.ndarray)
    assert isinstance(threshold, int) and 0 <= threshold <= 255, "Threshold value is not in the given range [0,255]"
    assert isinstance(newValue, int) and 0 <= newValue <= 255, "Maximum pixel value is not in the given range [0,255]"
    assert thresholdType in THRESHOLD_TYPES, "Invalid threshold type."
    return cv2.threshold(image, threshold, newValue, thresholdType)[1]


def rotateImage(image, angle):
    imageCenter = tuple(np.array(image.shape[1::-1]) / 2)
    rotMatrix = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)
    result = cv2.warpAffine(image, rotMatrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def emptyImage(shape: tuple) -> np.ndarray:
    """
    Creates zero (empty) image of the given shape.
    :param shape: shape of the image
    :return: zero matrix - image
    """
    return np.zeros(shape)


def cloneImage(image: np.ndarray) -> np.ndarray:
    """
    Makes a copy of the input image.
    :param image: image that should be copied
    :return: clone of the image.
    """
    return np.copy(image)


def applyCustomProcessingMethod(image, processingMethod, *args):
    obj = eval(processingMethod)
    return obj(image, *args)


# math operations on image
def convertTo3Channel(image: np.ndarray) -> np.ndarray:
    """
    Creates 3-channel alpha image
    :param image: input image
    :return: converted image
    """

    # if channelNo > 3:
    #     raise InvalidArgumentTypeOrValueError("Channel number is not currently supported.")
    return np.dstack([image] * 3)


def convertToSingleChannel(image: np.ndarray) -> np.ndarray:
    """
    Converts image to single-channel.
    :param image:  input image
    :return: converted image
    """
    return cast((image * 255), "uint8")


def cast(image: np.ndarray, type: str = "float32") -> np.ndarray:
    """
    Cast image to desired data type
    :param image: input image
    :param type:  target data type - currently supported datatypes are "float32" and "uint8"
    :return: converted image
    """
    if type not in ("float32", "uint8"):
        raise InvalidArgumentTypeOrValueError("Uknown target data type.")
    return image.astype(type)


def normalize(image: np.ndarray, normValue: float) -> object:
    if not isinstance(normValue, (int, float)):
        raise InvalidArgumentTypeOrValueError("Normalization value must be number.")
    return image / normValue


def blend(image: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    """
    Blends image and mask.
    :param image: image that will be blended
    :param mask: masking element
    :param color: mask color
    :return: blended image
    """
    return (mask * image) + ((1 - mask) * color)


def findEdgeNonBlackPixel(image, startEnd="start"):
    prev = None
    if startEnd == "start":
        for i in range(image.shape[0]):
            if sum(prev == np.zeros((1,3))[0]) == 3 and sum(image[i,0] == np.zeros((1,3))[0]) != 3:
                return i
            prev = image[i,0]
    else:
        for i in range(image.shape[0]-1, -1, -1):
            if sum(prev == np.zeros((1,3))[0]) == 3 and sum(image[i,0] == np.zeros((1,3))[0]) != 3:
                return i
            prev = image[i,0]
    return None
