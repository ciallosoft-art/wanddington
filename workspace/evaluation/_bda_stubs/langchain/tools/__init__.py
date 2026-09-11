"""`from langchain.tools import BaseTool, tool` must resolve at import time."""


class BaseTool:                      # subclassed at module scope in BDA's gene.py
    name: str = "stub"
    description: str = "stub"

    def _run(self, *a, **k):
        raise RuntimeError("langchain BaseTool stub invoked — install langchain for real tool use")

    async def _arun(self, *a, **k):
        raise RuntimeError("langchain BaseTool stub invoked — install langchain for real tool use")


def tool(*a, **k):                   # decorator form
    def _wrap(fn):
        return fn
    return _wrap(a[0]) if a and callable(a[0]) else _wrap
