"""Shared test helpers and fixtures."""

import socket


def find_free_port() -> int:
    """Return an OS-assigned ephemeral port that is currently free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
