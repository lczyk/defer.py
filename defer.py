"""
Single-file implementation of Go-like defer statement.

Based on https://habr.com/en/articles/191786/ by Denis Kolodin
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import sys
from functools import wraps
from types import CodeType, TracebackType
from typing import Any, Callable, Generator, TypeVar

Deferable = Callable[[], object]

_T = TypeVar("_T", bound=Callable[..., Any])
_D = TypeVar("_D", bound=Deferable)

__all__ = ["defer", "defers_collector"]

__version__ = "0.2.0"

log = logging.getLogger(__name__)


_WRAPPERS: set[CodeType] = set()
_COMPREHENSIONS = frozenset({"<listcomp>", "<setcomp>", "<dictcomp>"})
_CONTEXT_MANAGER = contextlib.contextmanager(lambda: None).__code__  # type: ignore
_ASYNC_CONTEXT_MANAGER = contextlib.asynccontextmanager(lambda: None).__code__  # type: ignore


def defer(x: _D) -> _D:
    """Defer a function call until the enclosing @defers_collector function exits.

    Must be called directly in the body of that function (list, set and dict
    comprehensions in the body count, generator expressions do not), otherwise raises
    RuntimeError. Returns ``x``, so it works as a decorator.
    """

    if not callable(x):
        raise TypeError(f"defer() argument must be callable, not {type(x).__name__}")

    frame = sys._getframe(1)
    # before 3.12 comprehensions run in their own frame: step out into the function
    # defining them. not generator expressions, those can outlive the call.
    while frame.f_code.co_name in _COMPREHENSIONS and frame.f_back is not None:
        frame = frame.f_back
    wrapper = frame.f_back
    if wrapper is None or wrapper.f_code not in _WRAPPERS:
        raise RuntimeError(
            "defer() must be called directly in a @defers_collector function, "
            f"not in {getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)}"
        )
    wrapper.f_locals["__defers__"].append(x)
    return x


class _DefersContainer:
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
                if inspect.isawaitable(res) and not _is_future(res):
                    if inspect.iscoroutine(res):
                        res.close()
                    raise TypeError(f"cannot await {d!r} outside an async function")
            except BaseException as e:  # noqa: PERF203
                interrupt = _caught(e, interrupt)
        if interrupt is not None and interrupt is not exc_value:
            try:
                raise interrupt
            finally:
                del interrupt  # break the frame <-> traceback cycle

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
            try:
                raise interrupt
            finally:
                del interrupt  # break the frame <-> traceback cycle


def _is_future(x: object) -> bool:
    # a future or task is already scheduled, so dropping it is fine. no future can
    # exist without asyncio loaded, so don't import it just to check
    asyncio = sys.modules.get("asyncio")
    return asyncio is not None and asyncio.isfuture(x)


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
    # a broken log handler must not stop the unwind
    with contextlib.suppress(Exception):
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
    code = getattr(func, "__code__", None)
    if code is _CONTEXT_MANAGER:
        # the collector has to wrap the generator itself, so re-apply in that order
        return contextlib.contextmanager(defers_collector(func.__wrapped__))  # type: ignore
    call = type(func).__call__  # a callable object is whatever its __call__ is
    if (
        code is _ASYNC_CONTEXT_MANAGER
        or inspect.isasyncgenfunction(func)
        or inspect.isasyncgenfunction(call)
    ):
        raise TypeError(
            "defers_collector does not support async generator functions, "
            "incl. under @asynccontextmanager"
        )

    if inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(call):

        @wraps(func)
        async def async_wrapped(*args: object, **kwargs: object) -> object:
            __defers__ = _DefersContainer()
            async with __defers__:
                return await func(*args, **kwargs)

        return _register(async_wrapped)  # type: ignore

    if inspect.isgeneratorfunction(func) or inspect.isgeneratorfunction(call):

        @wraps(func)
        def gen_wrapped(*args: object, **kwargs: object) -> Generator[Any, Any, Any]:
            __defers__ = _DefersContainer()
            with __defers__:
                return (yield from func(*args, **kwargs))

        return _register(gen_wrapped)  # type: ignore

    @wraps(func)
    def wrapped(*args: object, **kwargs: object) -> object:
        __defers__ = _DefersContainer()
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
