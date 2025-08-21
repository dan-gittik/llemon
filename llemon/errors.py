class Error(Exception):
    pass


class ConfigurationError(Error):
    pass


class InProgressError(Error):
    pass


class FinishedError(Error):
    pass


class IncompleteMessageError(Error):
    pass
