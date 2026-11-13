"""D-owned errors. The runner should record these as failed method runs."""


class ConfigurationError(ValueError):
    pass


class VerificationError(ValueError):
    pass


class RevisionError(ValueError):
    pass
