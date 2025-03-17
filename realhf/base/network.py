# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import socket
from contextlib import closing


def find_free_port(low=1, high=65536):
    """From stackoverflow Issue 1365265."""
    while True:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            if low <= port <= high:
                return port


def gethostname():
    return socket.gethostname()


def gethostip():
    return socket.gethostbyname(socket.gethostname())
