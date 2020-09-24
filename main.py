from img_proc import *
from constants import CANNY_THRESH_1, CANNY_THRESH_2, BLUR, MASK_DILATE_ITER, MASK_ERODE_ITER
from gesture_recognition import *
from cv2 import imread, imshow, waitKey, destroyAllWindows
from sys import argv
from io_utils import parseArguments
from exceptions import InvalidCommandLineArgsError, ImageNotFoundError

if __name__ == "__main__":
    # ---- read image -----
    if len(argv) < 2:
        raise InvalidCommandLineArgsError("Fatal error - image name must be provided")
    imagePath = argv[2]
    image = imread(imagePath)
    if image is None:
        raise ImageNotFoundError("Image file cannot be found. Check your image path.")

    arguments = argv[3:]
    parseArguments(arguments)

    # imshow("Original image", image)
    # waitKey(0)

    # ---- crop image and convert to grayscale -----

    if ImageParam.rotationAngle != 0:
        image = rotateImage(image, ImageParam.rotationAngle)
        # only for -90/90 deg rotations
        firstRow = findEdgeNonBlackPixel(image)
        if firstRow is not None:
            image = image[firstRow:, :]
        lastRow = findEdgeNonBlackPixel(image, "end")
        if lastRow is not None:
            image = image[:lastRow + 1, :]

    #if ImageParam.shouldTakeUpperHalf:
    if ImageParam.takeUpperHalf:
        image = getImageUpperPart(image, 1.5)  # parametrize this
        # imshow("Cropped image", image)
        # waitKey(0)
    gray = convertColorSpace(image, "grayscale")
    # imshow("Grayscale image", gray)
    # waitKey(0)

    # ----- edge detection and blurring-----
    edges = applyCannyEdge(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    #imshow("Edges", edges)
    #waitKey(0)

    edges = applyGaussianBlur(edges, (BLUR, BLUR))
    #imshow("Edges after bluring", edges)
    #waitKey(0)

    # ----- morphological processing -----
    edges = applyDilation(edges, None, iterations=1)
    #imshow("After dilation", edges)
    #waitKey(0)
    edges = applyErosion(edges, None, iterations=1)
    #imshow("After erosion", edges)
    #waitKey(0)

    # ----- Find contours -------
    contours = findImageContours(edges)
    maxContour = findMaxContour(contours)
    contourImage = cloneImage(image)
    drawImageContours(contourImage, contours, (255, 255, 255), 2)
    #imshow("Contours", contourImage)
    #waitKey(0)

    # ----- Masking -------
    mask = emptyImage(edges.shape)
    for c in contours:
        fillConvex(mask, c, 255)
    #imshow("Mask", mask)
    #waitKey(0)

    # do some processing on mask
    mask = applyDilation(mask, None, iterations=MASK_DILATE_ITER)
    mask = applyErosion(mask, None, iterations=MASK_ERODE_ITER)
    mask = applyGaussianBlur(mask, (BLUR, BLUR))
    #imshow("Processed mask", mask)
    #waitKey(0)
    binMask = convertToSingleChannel(mask)
    #imshow("Binary mask", binMask)
    #waitKey(0)

    imageCopy = cloneImage(image)
    predictGesture(maxContour, imageCopy)
    #waitUntilEnter()
    waitKey(0)
    destroyAllWindows()
