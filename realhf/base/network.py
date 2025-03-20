# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import socket
from contextlib import closing


def find_free_port(low=1, high=65536, exclude_ports=None):
    """Find a free port within the specified range, excluding certain ports."""
    if exclude_ports is None:
        exclude_ports = set()

    while True:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            if low <= port <= high and port not in exclude_ports:
                return port


def find_multiple_free_ports(count, low=1, high=65536):
    """Find multiple mutually exclusive free ports."""
    free_ports = set()
    for _ in range(count):
        port = find_free_port(low, high, exclude_ports=free_ports)
        free_ports.add(port)
    return list(free_ports)


def gethostname():
    return socket.gethostname()


def gethostip():
    return socket.gethostbyname(socket.gethostname())
