from exceptions import InvalidCommandLineArgsError
from image_param import ImageParam


# TODO: recode this - not elegant solution + move this to the separate file
def waitUntilEnter():
    input()

# easier to maintain
def parseArguments(arguments: list):
    argsMap = {}
    if len(arguments) % 2 == 1:
        raise InvalidCommandLineArgsError("Number of command line arguments must be even.")
    for i in range(0, len(arguments), 2):
        argName, argVal = arguments[i][2:], arguments[i + 1]
        try:argsMap[argName] = float(argVal)
        except (TypeError, ValueError) as e:
            if "," in argVal:
                argVal = tuple(argVal.split(","))
                argsMap[argName] = tuple(map(int, argVal))
    ImageParam.setParams(**argsMap)
