from hand_finger_detection import *
from img_draw import *
from img_contours import *
from img_proc import cloneImage
from cv2 import imshow, waitKey

from math_util import calculateLineSlope
from io_utils import waitUntilEnter


def findAndDrawFingerParams(maxContour, image):
    possibleFingersImage = cloneImage(image)
    keyPointsImage = cloneImage(image)
    convexHullImage = cloneImage(image)
    fingersImage = cloneImage(image)

    print("[1] Find defect points......", end=" ")
    defects = findConvexityDefects(maxContour)
    print("DONE")
    # print(f"{len(defects)} points found")
    print("[2] Get points from defects.......", end=" ")
    points = getConvexPoints(maxContour, defects)
    print("DONE")
    print(f"{len(points)} points found.")
    print("[3] Find root point.......", end=" ")
    rootPoint = findPalmRoot(points)
    print("DONE")
    print(f"Coordinates of the palm root point:{rootPoint}")
    print("[4] Finding close points.....", end=" ")
    closePoints = findClosePoints(points)
    print("DONE")
    print("[5] Merging close points.....", end=" ")
    mergedPoints = mergeClosePoints(closePoints)
    print("DONE")
    print(f"Merged points:{mergedPoints}")
    drawPoints(keyPointsImage, mergedPoints, color=(0, 255, 255))

    # drawCircle(keyPointsImage, rootPoint, 4, (255, 255, 255), -1)

    print("[6] Calculate distances between root point and other points........", end=" ")
    distances, farthestPoint = calculateRootToPointsDistances(rootPoint, mergedPoints)
    print("DONE")
    print(f"Coordinates of the farthest point:{farthestPoint}")

    for point in distances:
        drawLine(keyPointsImage, point, rootPoint, (0, 255, 0))
    drawPoints(keyPointsImage, [farthestPoint], color=(128, 128, 255))
    c, radius = findMinEnclosingCircle(maxContour)
    drawCircle(keyPointsImage, c, radius, color=(255, 0, 255))
    drawPoint(keyPointsImage, rootPoint, color=(255, 255, 255))

    fingers = mergedPoints

    possibleFingers = []
    print("[7] Remove low-positioned points (if any)........", end=" ")
    angleOffset = ImageParam.angleOffset
    cutoffAngles = ImageParam.cutoffAngles
    for point in fingers:
        angle = calculateLineSlope(rootPoint, point)
        angle += angleOffset
        if cutoffAngles[0] <= angle <= cutoffAngles[1]:
            continue
        possibleFingers.append(point)
    print("DONE")
    # possibleFingers = rotateContourPoints(possibleFingers)
    print(f"Fingers points afer removal operation: {possibleFingers}")
    print("[8] Remove collinear points........", end=" ")
    possibleFingers = removeCollinearFingers(possibleFingers)
    drawPoints(possibleFingersImage, possibleFingers, color=(0, 0, 255))
    print("DONE")

    # remove points from distance dict
    distancesCopy = distances.copy()
    for finger in distancesCopy:
        # TODO: should be optimized
        if finger not in possibleFingers:
            distances.pop(finger)
    # print(len(fings))
    foundFingers = possibleFingers
    print("[9] Remove false fingers (closest to root point)........", end=" ")
    if len(distances) > 5:
        foundFingers = removeFalseFingers(distances)
    print("DONE")

    drawCircle(convexHullImage, farthestPoint, 5, (0, 255, 255), -1)
    drawPoints(fingersImage, foundFingers, color=(255, 0, 0))

    defectPoints = getConvexPoints(maxContour, defects, "all")
    print("[10] Find finger valleys........", end=" ")
    valleys = findFingerValleys(defectPoints)
    print("DONE")
    print(f"{len(valleys)} finger valleys found - {valleys}")
    drawConvexHull(convexHullImage, defectPoints)
    if len(valleys) > 0:
        point = valleys[0][2]
        drawCircle(keyPointsImage, point, 4, (0, 0, 200), -1)
    print("Press any key to show useful images with calculated and found parameters.")
    waitUntilEnter()
    imshow("Convex hull", convexHullImage)
    imshow("Key points", keyPointsImage)
    imshow("Possible fingers", possibleFingersImage)
    # imshow("Fingers", fingersImage)
    return fingersImage, foundFingers, rootPoint, distances, valleys


def determineGesture(image, fingers, rootPoint, rootToPointsDistances, valleys):
    decisionCredibility = "uncertain"
    foundGesture = None
    maxDistance = max(rootToPointsDistances.values())
    minDistance = min(rootToPointsDistances.values())
    sortedFingersByHeight = sorted(fingers, key=lambda coord: rootPoint[1] - coord[1], reverse=True)
    if len(valleys) == 1:
        if len(fingers) == 2:
            foundGesture = "scissors"
            decisionCredibility = "certain"
        elif len(fingers) == 5:
            foundGesture = "paper"
            decisionCredibility = "certain"
        else:
            a = distanceBetweenPoints(valleys[0][0], valleys[0][2])
            b = distanceBetweenPoints(valleys[0][1], valleys[0][2])
            valley = valleys[0]
            if ((abs(valley[0][1] - valley[1][1]) < ImageParam.valleyTopPointsHeightDiff) and
                valley[2][1] - max(valley[0][1], valley[1][1]) > ImageParam.valleyDepthThreshold) or \
                    (a >= ImageParam.valleySidesLengthFactor * b) or \
                    sortedFingersByHeight[0][1] - sortedFingersByHeight[-1][0] > ImageParam.fingersHeightDiff \
                    or (valley[2][1] - valley[0][1] > ImageParam.valleyDepthThreshold and
                        valley[2][1] - valley[1][1]) > ImageParam.valleyDepthThreshold:
                foundGesture = "scissors"
                decisionCredibility = "certain"
            else:
                foundGesture = "paper"
                decisionCredibility = "certain"
    else:
        if len(fingers) < 5:
            foundGesture = "rock"
            decisionCredibility = "certain"
        else:
            # PAPER OR ROCK
            minHeightFinger = min(fingers, key=lambda x: x[1])
            maxHeightFinger = max(fingers, key=lambda x: x[1])

            fingerLengths = [rootPoint[1] - finger[1] for finger in sortedFingersByHeight[:3]]
            if len(valleys) > 0:
                if len(fingers) == 5 or (maxHeightFinger[1] - minHeightFinger[1] > ImageParam.fingersLengthDiff) and (
                        maxDistance - minDistance > ImageParam.minMaxFingerDist):
                    foundGesture = "paper"
                    if all([finger >= ImageParam.longestFingersHeightsThreshold for finger in fingerLengths]):
                        decisionCredibility = "certain"
                    else:
                        decisionCredibility = "uncertain - rock properties"
            else:
                if (maxHeightFinger[1] - minHeightFinger[1] > ImageParam.fingersLengthDiff) and (
                        maxDistance - minDistance > ImageParam.minMaxFingerDist) and \
                        all([fl >= ImageParam.longestFingersHeightsThreshold for fl in fingerLengths]):
                    decisionCredibility = "uncertain - rock properties"
                    foundGesture = "paper"
                else:
                    foundGesture = "rock"
                    decisionCredibility = "uncertain - paper properties"
    return foundGesture, decisionCredibility


def predictGesture(maxContour, image):
    fingersImage, foundFingers, rootPoint, distances, valleys = findAndDrawFingerParams(maxContour, image)
    gesture, decisionCredibility = determineGesture(fingersImage, foundFingers, rootPoint, distances, valleys)
    print(f"Gesture on the image: {gesture}. Decision credibility = {decisionCredibility}.")
    drawText(fingersImage, gesture)
    imshow("Fingers", fingersImage)
