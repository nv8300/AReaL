from enum import Enum


class ReturnFormat(Enum):
    """
    ReturnFormat controls the decode_ast logic.
    """
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSON = "json"
    VERBOSE_XML = "verbose_xml"
    CONCISE_XML = "concise_xml"