from _stub import _Unavailable


def __getattr__(item):
    return _Unavailable("paperqa." + item)
