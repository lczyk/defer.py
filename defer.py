"""
Single-file implementation of Go-like defer statement.

Based on https://habr.com/en/articles/191786/ by Denis Kolodin
"""

import inspect
import sys
import traceback
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import TypeVar

Deferable = Callable[[], object]

_T = TypeVar("_T", bound=Callable[..., object])

__all__ = ["defer", "defers_collector"]

__version__ = "0.1.2"


def defer(x: Deferable) -> None:
    """Defer a function call until the current function scope exits."""

    if not callable(x):
        raise TypeError(f"defer() argument must be callable, not {type(x).__name__}")

    for f in inspect.stack():
        if "__defers__" in f[0].f_locals:
            f[0].f_locals["__defers__"].append(x)
            break


class DefersContainer:
    def __init__(self) -> None:
        self.defers: list[Deferable] = []

    def append(self, defer: Deferable) -> None:
        self.defers.append(defer)

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # pop rather than iterate, so defers registered by a running defer still run
        interrupt = None
        while self.defers:
            d = self.defers.pop()
            try:
                d()
            except Exception as e:  # noqa: PERF203
                _report(e)
            except BaseException as e:
                # KeyboardInterrupt, SystemExit, ...: finish unwinding, then propagate
                if interrupt is None:
                    interrupt = e
                else:
                    _report(e)
        if interrupt is not None:
            raise interrupt


def _report(e: BaseException) -> None:
    # runs mid-unwind, so a broken stderr must not replace the real exception
    try:
        if sys.stderr is None:
            return
        sys.stderr.write("Error in defer:\n" + "".join(traceback.format_exception(e)))
        sys.stderr.flush()
    except Exception:
        pass


def defers_collector(func: _T) -> _T:
    """Marks a function to collect defers."""

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapped(*args: object, **kwargs: object) -> object:
            __defers__ = DefersContainer()
            with __defers__:
                return await func(*args, **kwargs)

        return async_wrapped  # type: ignore

    @wraps(func)
    def wrapped(*args: object, **kwargs: object) -> object:
        __defers__ = DefersContainer()
        with __defers__:
            return func(*args, **kwargs)

    return wrapped  # type: ignore


if __name__ == "__main__":

    @defers_collector
    def func() -> None:
        print("Start")
        defer(lambda: print("Defer called!"))

        print("End")

    func()
