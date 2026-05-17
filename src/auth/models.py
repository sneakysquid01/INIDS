from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    """Immutable result of a successful authentication check."""

    user_id: str
    username: str
    roles: frozenset[str]


class AuthError(Exception):
    """Raised by auth layer; carries HTTP status code."""

    def __init__(self, message: str = "Unauthorized", status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code
