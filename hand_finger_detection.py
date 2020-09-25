from math_util import distanceBetweenPoints, calculateAngle, calculateLineSlope
from numpy import mean, median, pi
from image_param import ImageParam


def findClosePoints(points):
    closePoints = {}
    pointNo = 0
    batch = []
    distThreshold = ImageParam.consecutivePointsDistThreshold
    coordsOffset = ImageParam.consecutivePointsCoordOffset
    for i in range(len(points) - 1):
        dist = distanceBetweenPoints(points[i], points[i + 1])
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if dist < distThreshold or (abs(x1 - x2) <= coordsOffset and abs(y1 - y2) <= coordsOffset):
            batch.append(points[i])
        else:
            if batch:
                batch.append(points[i])
                closePoints[pointNo] = batch
            else:
                closePoints[pointNo] = [points[i]]
            batch = []
            pointNo += 1
    return closePoints


def mergeClosePoints(closePoints):
    mergedPoints = []
    for pointNo, clPoints in closePoints.items():
        if len(clPoints) == 1:
            mergedPoints.append(clPoints[0])
        else:
            medianX = int(median([val[0] for val in clPoints]))
            medianY = int(median([val[1] for val in clPoints]))
            mergedPoints.append((medianX, medianY))
    return mergedPoints


def findLowestPoints(points):
    sortedPointsByY = sorted(points, key=lambda x: x[1], reverse=True)
    # TODO: add validation - if len(points) < 2
    return sortedPointsByY[0], sortedPointsByY[1]


def findPalmRoot(points):
    if len(points) < 2:
        raise Exception("Fatal error.")
    y = max(points, key=lambda point: point[1])[1]  # ymax
    pointsSortedByX = sorted(points, key=lambda point: point[0])
    x = (pointsSortedByX[0][0] + pointsSortedByX[-1][0]) // 2
    return x, y


def calculateRootToPointsDistances(rootPoint, points):
    pointsDist = {}
    maxDist = 0
    farthestPoint = None
    for point in points:
        dist = distanceBetweenPoints(rootPoint, point)
        pointsDist[point] = dist
        maxDist = max(maxDist, dist)
        if maxDist == dist: farthestPoint = point
    return pointsDist, farthestPoint


def findFingerValleys(defectPoints):
    valleys = []
    for defect in defectPoints:
        start, end, far = defect
        a = distanceBetweenPoints(start, end)
        b = distanceBetweenPoints(start, far)
        c = distanceBetweenPoints(end, far)
        angle = calculateAngle(a, b, c)
        if angle <= pi / 2 and (start[1] < far[1] and end[1] < far[1]):
            valleys.append((start, end, far))

    return valleys


def rotateContourPoints(points):
    for i in range(len(points) - 1):
        if points[i + 1][0] > points[i][0]:
            break
    if i == len(points) - 1:
        return points
    return points[i + 1:] + points[:i + 1]


def removeCollinearFingers(fingers):
    fingerSet = set(fingers)
    angleOffset = ImageParam.angleOffset
    for i in range(len(fingers) - 1):
        angle = (calculateLineSlope(fingers[i], fingers[i + 1]) + angleOffset) % 180
        angleBounds = ImageParam.angleBounds
        if angleBounds[0] <= angle <= angleBounds[1]:
            if fingers[i][1] <= fingers[i + 1][1]:
                fingerToRemove = fingers[i+1]
            else:
                fingerToRemove = fingers[i]
            if fingerToRemove in fingerSet:
                fingerSet.remove(fingerToRemove)
    return list(fingerSet)


def removeFalseFingers(distances, fingersNo=5):
    fingers = []
    for finger, dist in sorted(distances.items(), key=lambda x: x[1], reverse=True):
        fingers.append(finger)
    return fingers[:fingersNo]
