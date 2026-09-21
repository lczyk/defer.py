"""
Single-file implementation of Go-like defer statement.

Based on https://habr.com/en/articles/191786/ by Denis Kolodin
"""

import contextlib
import inspect
import logging
import sys
from collections.abc import Callable, Generator
from functools import wraps
from types import CodeType, TracebackType
from typing import Any, TypeVar

Deferable = Callable[[], object]

_T = TypeVar("_T", bound=Callable[..., Any])

__all__ = ["defer", "defers_collector"]

__version__ = "0.1.2"

log = logging.getLogger(__name__)


_WRAPPERS: set[CodeType] = set()
_COMPREHENSIONS = frozenset({"<listcomp>", "<setcomp>", "<dictcomp>", "<genexpr>"})
_CONTEXT_MANAGERS: dict[CodeType, Callable[[Any], Any]] = {
    cm(lambda: None).__code__: cm  # type: ignore
    for cm in (contextlib.contextmanager, contextlib.asynccontextmanager)
}


def defer(x: Deferable) -> None:
    """Defer a function call until the enclosing @defers_collector function exits.

    Must be called directly in the body of that function (comprehensions in the body
    count), otherwise raises RuntimeError.
    """

    if not callable(x):
        raise TypeError(f"defer() argument must be callable, not {type(x).__name__}")

    frame = sys._getframe(1)
    # comprehensions run in their own frame before 3.12, generator expressions always
    # do. step out only into the function defining them, not whoever drives a genexpr.
    while frame.f_code.co_name in _COMPREHENSIONS:
        parent = frame.f_back
        if parent is None or not any(
            c is frame.f_code for c in parent.f_code.co_consts
        ):
            break
        frame = parent
    wrapper = frame.f_back
    if wrapper is None or wrapper.f_code not in _WRAPPERS:
        raise RuntimeError(
            "defer() must be called directly in a @defers_collector function, "
            f"not in {frame.f_code.co_qualname}"
        )
    wrapper.f_locals["__defers__"].append(x)


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
        interrupt = _in_flight(exc_value)
        while self.defers:
            d = self.defers.pop()
            try:
                res = d()
                if inspect.isawaitable(res):
                    if inspect.iscoroutine(res):
                        res.close()
                    raise TypeError(f"cannot await {d!r} outside an async function")
            except BaseException as e:  # noqa: PERF203
                interrupt = _caught(e, interrupt)
        if interrupt is not None and interrupt is not exc_value:
            raise interrupt

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        interrupt = _in_flight(exc_value)
        while self.defers:
            d = self.defers.pop()
            try:
                res = d()
                if inspect.isawaitable(res):
                    await res
            except BaseException as e:  # noqa: PERF203
                interrupt = _caught(e, interrupt)
        if interrupt is not None and interrupt is not exc_value:
            raise interrupt


def _in_flight(exc: BaseException | None) -> BaseException | None:
    # an interrupt already propagating outranks any a defer raises
    return None if exc is None or isinstance(exc, Exception) else exc


def _caught(e: BaseException, interrupt: BaseException | None) -> BaseException | None:
    # KeyboardInterrupt, SystemExit, ...: finish unwinding, then propagate the first
    if isinstance(e, Exception) or interrupt is not None:
        _report(e)
        return interrupt
    return e


def _report(e: BaseException) -> None:
    log.error("Error in defer", exc_info=e)


def defers_collector(func: _T) -> _T:
    """Marks a function to collect defers, which run LIFO when it exits.

    Works on plain, async and generator functions, and on either side of
    @contextlib.contextmanager. Not above @property; no async generator functions.

    Exceptions from deferred calls are logged. KeyboardInterrupt, SystemExit and
    other non-Exception errors propagate once every deferred call has run.
    """

    if isinstance(func, (staticmethod, classmethod)):
        return type(func)(defers_collector(func.__func__))
    if isinstance(func, property):
        raise TypeError("defers_collector must be applied below @property, not above")
    wrap_cm = _CONTEXT_MANAGERS.get(getattr(func, "__code__", None))  # type: ignore[arg-type]
    if wrap_cm is not None:
        # the collector has to wrap the generator itself, so re-apply in that order
        return wrap_cm(defers_collector(func.__wrapped__))  # type: ignore
    if inspect.isasyncgenfunction(func):
        raise TypeError("defers_collector does not support async generator functions")

    if inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
        type(func).__call__
    ):

        @wraps(func)
        async def async_wrapped(*args: object, **kwargs: object) -> object:
            __defers__ = DefersContainer()
            async with __defers__:
                return await func(*args, **kwargs)

        return _register(async_wrapped)  # type: ignore

    if inspect.isgeneratorfunction(func):

        @wraps(func)
        def gen_wrapped(*args: object, **kwargs: object) -> Generator[Any, Any, Any]:
            __defers__ = DefersContainer()
            with __defers__:
                return (yield from func(*args, **kwargs))

        return _register(gen_wrapped)  # type: ignore

    @wraps(func)
    def wrapped(*args: object, **kwargs: object) -> object:
        __defers__ = DefersContainer()
        with __defers__:
            return func(*args, **kwargs)

    return _register(wrapped)  # type: ignore


def _register(wrapper: Callable[..., Any]) -> Callable[..., Any]:
    # each kind of wrapper shares one code object across decorations, so this stays tiny
    _WRAPPERS.add(wrapper.__code__)
    return wrapper


if __name__ == "__main__":

    @defers_collector
    def func() -> None:
        print("Start")
        defer(lambda: print("Defer called!"))

        print("End")

    func()
