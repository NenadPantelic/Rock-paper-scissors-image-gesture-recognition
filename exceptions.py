class InvalidArgumentTypeOrValueError(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidCommandLineArgsError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ImageNotFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)
