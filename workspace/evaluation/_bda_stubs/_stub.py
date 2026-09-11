"""Shared machinery: a module object whose every attribute raises when actually used."""


class _Unavailable:
    def __init__(self, name: str) -> None:
        self.__name = name

    def _fail(self, *a, **k):
        raise RuntimeError(
            f"{self.__name} is a stub installed for the controlled BioDiscoveryAgent run "
            f"(no-tools configuration). Reaching it means a tool was actually invoked; "
            f"install the real dependency instead of trusting this result."
        )

    __call__ = _fail
    def __getattr__(self, item):
        return _Unavailable(f"{self.__name}.{item}")


def stub_module(name: str):
    import sys
    import types
    mod = types.ModuleType(name)

    def __getattr__(item, _n=name):
        return _Unavailable(f"{_n}.{item}")

    mod.__getattr__ = __getattr__
    mod.__path__ = []
    sys.modules[name] = mod
    return mod
