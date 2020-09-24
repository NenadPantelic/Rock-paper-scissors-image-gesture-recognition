from exceptions import ImageNotFoundError
from img_proc import *
from constants import CANNY_THRESH_1, CANNY_THRESH_2, BLUR, MASK_DILATE_ITER, MASK_ERODE_ITER
from gesture_recognition import *
from cv2 import imread, imshow, waitKey, destroyAllWindows
from sys import argv
from io_utils import parseArguments

"""
FOR TEST ONLY
"""
imagePath = "images/paper_1.jpg"
imagePath = "images/paper_2.jpg"
imagePath = "images/paper_3.jpg"
imagePath = "images/paper_4.jpg"  # --angleOffset 30
imagePath = "images/paper_5.jpg"
imagePath = "images/paper_6.jpg"  # --angleOffset -15
imagePath = "images/paper_7.jpg"  # --takeUpperHalf 1
imagePath = "images/paper_8.jpg"
imagePath = "images/paper_9.jpg"
imagePath = "images/paper_10.jpg"
imagePath = "images/paper_11.jpg"  # --rotationAngle 90 --takeUpperHalf 1
imagePath = "images/paper_12.jpg"  # --angleOffset 45
imagePath = "images/paper_13.jpg"  # --rotationAngle  -90

###  ROCK ###

imagePath = "images/rock_1.jpg"
imagePath = "images/rock_2.jpg"
imagePath = "images/rock_3.jpg"
imagePath = "images/rock_4.jpg"  # --rotationAngle 90
imagePath = "images/rock_5.jpg"
imagePath = "images/rock_6.jpg"  # --rotationAngle 90
imagePath = "images/rock_7.jpg"
imagePath = "images/rock_8.jpg"
imagePath = "images/rock_9.jpg"
imagePath = "images/rock_10.jpg"
imagePath = "images/rock_11.jpg"  # palac istaknut --takeUpperHalf
imagePath = "images/rock_12.jpg"  # prst uvucen malo --takeUpperHalf
imagePath = "images/rock_13.jpg"

### SCISSORS ###
imagePath = "images/scissors_1.jpg"
imagePath = "images/scissors_2.jpg"  # --rotationAngle -90 --angleOffset 15
imagePath = "images/scissors_3.jpg"  # ne radi savrseno
imagePath = "images/scissors_4.jpg"
imagePath = "images/scissors_5.jpg"
imagePath = "images/scissors_6.jpg"
imagePath = "images/scissors_7.jpg"
imagePath = "images/scissors_8.jpg"  # --angleOffset 50 --takeUpperHalf 1
imagePath = "images/scissors_9.jpg"
imagePath = "images/scissors_10.jpg"
imagePath = "images/scissors_11.jpg"
imagePath = "images/scissors_12.jpg"
imagePath = "images/scissors_13.jpg"  # --rotationAngle 90

if __name__ == "__main__":
    # ---- read image -----
    arguments = argv[1:]
    parseArguments(arguments)

    image = imread(imagePath)
    if image is None:
        raise ImageNotFoundError("Image file cannot be found. Check your image path.")

    if ImageParam.rotationAngle != 0:
        image = rotateImage(image, ImageParam.rotationAngle)
        # only for -90/90 deg rotations
        firstRow = findEdgeNonBlackPixel(image)
        if firstRow is not None:
            image = image[firstRow:, :]
        lastRow = findEdgeNonBlackPixel(image, "end")
        if lastRow is not None:
            image = image[:lastRow + 1, :]

    # imshow("Original image", image)
    # waitKey(0)

    # ---- crop image and convert to grayscale -----
    if ImageParam.takeUpperHalf:
        image = getImageUpperPart(image, 1.5)  # parametrize this

    # imshow("Cropped image", image)
    # waitKey(0)
    gray = convertColorSpace(image, "grayscale")
    # imshow("Grayscale image", gray)
    # waitKey(0)

    # ----- edge detection and blurring-----
    edges = applyCannyEdge(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    # imshow("Edges", edges)
    # waitKey(0)

    edges = applyGaussianBlur(edges, (BLUR, BLUR))
    # imshow("Edges after bluring", edges)
    # waitKey(0)

    # ----- morphological processing -----
    edges = applyDilation(edges, None, iterations=1)
    # imshow("After dilation", edges)
    # waitKey(0)
    edges = applyErosion(edges, None, iterations=1)
    # imshow("After erosion", edges)
    # waitKey(0)

    # ----- Find contours -------
    # #contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = findImageContours(edges)
    maxContour = findMaxContour(contours)
    contourImage = cloneImage(image)
    drawImageContours(contourImage, contours, (255, 255, 255), 2)
    # imshow("Contours", contourImage)
    # waitKey(0)

    # ----- Masking -------
    mask = emptyImage(edges.shape)
    for c in contours:
        fillConvex(mask, c, 255)
    # imshow("Mask", mask)
    # waitKey(0)

    # do some processing on mask
    mask = applyDilation(mask, None, iterations=MASK_DILATE_ITER)
    mask = applyErosion(mask, None, iterations=MASK_ERODE_ITER)
    mask = applyGaussianBlur(mask, (BLUR, BLUR))
    # imshow("Processed mask", mask)
    # waitKey(0)
    binMask = convertToSingleChannel(mask)
    # imshow("Binary mask", binMask)
    # waitKey(0)

    imageCopy = cloneImage(image)
    predictGesture(maxContour, imageCopy)
    # waitUntilEnter()
    waitKey(0)
    destroyAllWindows()
