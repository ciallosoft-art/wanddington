from _stub import _Unavailable


def __getattr__(item):
    return _Unavailable("html2text." + item)
