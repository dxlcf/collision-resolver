from __future__ import annotations


class CollisionResolverError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message


class ValidationError(CollisionResolverError):
    pass


class BackendUnavailableError(CollisionResolverError):
    pass
