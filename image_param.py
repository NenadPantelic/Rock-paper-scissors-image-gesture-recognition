class ImageParam:
    # angle params
    angleBounds = (75, 105)  # TODO: lower-upper validation
    angleOffset = 0
    cutoffAngles = (-15, 15)  # TODO: lower-upper validation
    rotationAngle = 0

    # flags
    #shouldTakeUpperHalf = False
    takeUpperHalf = False

    # dist params
    consecutiveDefectPointsDist = 30.0

    # coordinates offsets
    consecutiveDefectPointsCoordOffset = 5.0

    # valley params
    valleyTopPointsHeightDiff = 50
    valleyDepthThreshold = 70
    valleySidesLengthFactor = 1.2

    # finger params
    fingersHeightDiff = 100
    fingersLengthDiff = 140
    minMaxFingerDist = 100
    longestFingersHeightsThreshold = 280

    @staticmethod
    def setParams(**params):
        for key, value in params.items():
            if hasattr(ImageParam, key):
                ImageParam.key = value
                setattr(ImageParam, key, value)
