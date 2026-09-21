import asyncio
import contextlib
import gc
import inspect
import logging
import subprocess
import sys
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Coroutine, Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from defer import defer, defers_collector

HERE = Path(__file__).parent


def test_basic() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("deferred"))
        out.append("body")

    f()
    assert out == ["body", "deferred"]


def test_defer_order() -> None:
    out: list[int] = []

    @defers_collector
    def f() -> None:
        for j in range(10):
            defer(lambda j=j: out.append(j))

    f()
    assert out == list(reversed(range(10)))


def test_fires_on_every_call() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("R"))

    f()
    f()
    f()
    assert out == ["R", "R", "R"]


def test_deferred_sees_state_at_fire_time() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        state = {"msg": ""}
        defer(lambda: out.append(state["msg"]))
        state["msg"] = "hello world"

    f()
    assert out == ["hello world"]


def test_late_binding_closure() -> None:
    # python closure semantics, not ours: without the default-arg trick every
    # deferred lambda sees the loop variable's final value
    out: list[int] = []

    @defers_collector
    def f() -> None:
        for j in range(3):
            defer(lambda: out.append(j))  # noqa: B023

    f()
    assert out == [2, 2, 2]


@pytest.mark.parametrize("kind", ["list", "set", "dict", "genexpr"])
def test_defer_in_comprehension(kind: str) -> None:
    out: list[int] = []

    @defers_collector
    def f() -> None:
        fns = [lambda i=i: out.append(i) for i in range(3)]
        if kind == "list":
            [defer(fn) for fn in fns]
        elif kind == "set":
            {defer(fn) for fn in fns}
        elif kind == "dict":
            {i: defer(fn) for i, fn in enumerate(fns)}
        else:
            list(defer(fn) for fn in fns)

    f()
    assert out == [2, 1, 0]


def test_defer_in_nested_comprehension() -> None:
    out: list[tuple[int, int]] = []

    @defers_collector
    def f() -> None:
        [
            [defer(lambda i=i, j=j: out.append((i, j))) for j in range(2)]
            for i in range(2)
        ]

    f()
    assert out == [(1, 1), (1, 0), (0, 1), (0, 0)]


def test_comprehension_in_undecorated_helper_raises() -> None:
    def helper() -> None:
        [defer(lambda: None) for _ in range(1)]

    @defers_collector
    def f() -> None:
        helper()

    with pytest.raises(RuntimeError, match="helper"):
        f()


def test_genexpr_consumed_elsewhere_raises() -> None:
    @defers_collector
    def consume(gen: Iterator[None]) -> None:
        list(gen)

    @defers_collector
    def f() -> None:
        consume(defer(lambda: None) for _ in range(1))

    with pytest.raises(RuntimeError, match=r"f\.<locals>\.<genexpr>"):
        f()


def test_defer_in_undecorated_helper_raises() -> None:
    def helper() -> None:
        defer(lambda: None)

    @defers_collector
    def f() -> None:
        helper()

    with pytest.raises(RuntimeError, match=r"not in \w+\.<locals>\.helper$"):
        f()


def test_escaped_closure_raises() -> None:
    out: list[str] = []

    @defers_collector
    def make() -> Callable[[], None]:
        return lambda: defer(lambda: out.append("escaped"))

    late = make()

    @defers_collector
    def other() -> None:
        late()

    with pytest.raises(RuntimeError):
        other()
    assert out == []


def test_local_named_defers_is_not_a_collector() -> None:
    def f() -> list[object]:
        __defers__: list[object] = []
        g()
        return __defers__

    def g() -> None:
        defer(lambda: None)

    with pytest.raises(RuntimeError):
        f()


def test_decorator_between_collector_and_function_raises() -> None:
    def passthrough(func: Callable[[], None]) -> Callable[[], None]:
        def inner() -> None:
            func()

        return inner

    @defers_collector
    @passthrough
    def f() -> None:
        defer(lambda: None)

    with pytest.raises(RuntimeError):
        f()


def test_nested_collectors() -> None:
    out: list[str] = []

    @defers_collector
    def inner() -> None:
        defer(lambda: out.append("inner"))

    @defers_collector
    def outer() -> None:
        defer(lambda: out.append("outer"))
        inner()
        out.append("outer body")

    outer()
    assert out == ["inner", "outer body", "outer"]


def test_recursion() -> None:
    out: list[str] = []

    @defers_collector
    def f(n: int) -> None:
        defer(lambda: out.append(f"exit {n}"))
        out.append(f"enter {n}")
        if n > 0:
            f(n - 1)

    f(2)
    assert out == ["enter 2", "enter 1", "enter 0", "exit 0", "exit 1", "exit 2"]


def test_passes_args_and_kwargs() -> None:
    got: list[object] = []

    @defers_collector
    def f(a: int, b: int, *, c: int) -> None:
        got.extend([a, b, c])

    f(1, 2, c=3)
    assert got == [1, 2, 3]


def test_preserves_metadata() -> None:
    @defers_collector
    def my_func() -> None:
        """My docstring."""

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."
    assert my_func.__wrapped__ is not None  # type: ignore[attr-defined]


def test_works_on_methods() -> None:
    out: list[str] = []

    class C:
        @defers_collector
        def m(self) -> None:
            defer(lambda: out.append("deferred"))
            out.append("body")

    C().m()
    assert out == ["body", "deferred"]


@pytest.mark.parametrize("kind", [staticmethod, classmethod])
def test_decorator_above_static_and_class_method(kind: type) -> None:
    out: list[str] = []

    def m(*_: object) -> str:
        defer(lambda: out.append("deferred"))
        out.append("body")
        return "value"

    class C:
        meth = defers_collector(kind(m))

    assert isinstance(C.__dict__["meth"], kind)
    assert C.meth() == "value"
    assert C().meth() == "value"
    assert out == ["body", "deferred"] * 2


def test_decorator_above_async_staticmethod() -> None:
    out: list[str] = []

    class C:
        @defers_collector
        @staticmethod
        async def m() -> None:
            defer(lambda: out.append("deferred"))
            await asyncio.sleep(0)
            out.append("body")

    asyncio.run(C().m())
    assert out == ["body", "deferred"]


def test_decorator_above_property_raises() -> None:
    with pytest.raises(TypeError, match="below @property"):

        class C:
            @defers_collector  # type: ignore[arg-type]
            @property
            def x(self) -> int:
                return 1


def test_async_callable_object() -> None:
    out: list[str] = []

    class Handler:
        async def __call__(self) -> str:
            defer(lambda: out.append("deferred"))
            await asyncio.sleep(0)
            out.append("body")
            return "value"

    f = defers_collector(Handler())
    assert inspect.iscoroutinefunction(f)
    assert asyncio.run(f()) == "value"
    assert out == ["body", "deferred"]


def test_async_generator_function_raises() -> None:
    with pytest.raises(TypeError, match="async generator"):

        @defers_collector
        async def agen() -> AsyncIterator[int]:
            yield 1


def test_handoff_with_flag() -> None:
    # go idiom: the cleanup only runs if ownership was not handed to the caller
    released: list[str] = []

    @defers_collector
    def acquire(fail: bool) -> str:
        handed_off = False
        defer(lambda: handed_off or released.append("resource"))
        if fail:
            raise ValueError
        handed_off = True
        return "resource"

    assert acquire(fail=False) == "resource"
    assert released == []
    with pytest.raises(ValueError):
        acquire(fail=True)
    assert released == ["resource"]


def test_no_defers() -> None:
    called = []

    @defers_collector
    def f() -> None:
        called.append(True)

    f()
    assert called == [True]


def test_defer_outside_collector_raises() -> None:
    with pytest.raises(RuntimeError, match="must be called directly"):
        defer(lambda: None)


def test_no_reference_cycle() -> None:
    class Big:
        pass

    ref: weakref.ref[Big] | None = None

    @defers_collector
    def f() -> None:
        nonlocal ref
        big = Big()
        ref = weakref.ref(big)
        defer(lambda: None)

    gc.disable()
    try:
        f()
        assert ref is not None
        assert ref() is None
    finally:
        gc.enable()


################################################################################
# generators


def test_generator_defers_run_on_exhaustion() -> None:
    out: list[str] = []

    @defers_collector
    def gen() -> Iterator[int]:
        defer(lambda: out.append("first"))
        yield 1
        defer(lambda: out.append("second"))
        yield 2

    it = gen()
    assert out == []
    assert list(it) == [1, 2]
    assert out == ["second", "first"]


def test_generator_defers_run_on_close() -> None:
    out: list[str] = []

    @defers_collector
    def gen() -> Iterator[int]:
        defer(lambda: out.append("deferred"))
        yield 1
        out.append("never")
        yield 2

    it = gen()
    next(it)
    it.close()
    assert out == ["deferred"]


def test_generator_body_exception() -> None:
    out: list[str] = []

    @defers_collector
    def gen() -> Iterator[int]:
        defer(lambda: out.append("deferred"))
        yield 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(gen())
    assert out == ["deferred"]


def test_generator_send_and_return_value() -> None:
    out: list[str] = []

    @defers_collector
    def gen() -> Generator[int, int, str]:
        defer(lambda: out.append("deferred"))
        got = yield 1
        return f"got {got}"

    it = gen()
    next(it)
    assert out == []
    with pytest.raises(StopIteration) as info:
        it.send(42)
    assert info.value.value == "got 42"
    assert out == ["deferred"]


def _cm_below(out: list[str]) -> Callable[[], contextlib.AbstractContextManager[str]]:
    @contextlib.contextmanager
    @defers_collector
    def cm() -> Iterator[str]:
        defer(lambda: out.append("deferred"))
        out.append("enter")
        yield "value"
        out.append("exit")

    return cm


def _cm_above(out: list[str]) -> Callable[[], contextlib.AbstractContextManager[str]]:
    @defers_collector
    @contextlib.contextmanager
    def cm() -> Iterator[str]:
        defer(lambda: out.append("deferred"))
        out.append("enter")
        yield "value"
        out.append("exit")

    return cm


@pytest.mark.parametrize("make", [_cm_below, _cm_above], ids=["below", "above"])
def test_contextmanager(
    make: Callable[[list[str]], Callable[[], contextlib.AbstractContextManager[str]]],
) -> None:
    out: list[str] = []
    cm = make(out)
    with cm() as value:
        assert value == "value"
        assert out == ["enter"]
    assert out == ["enter", "exit", "deferred"]


@pytest.mark.parametrize("make", [_cm_below, _cm_above], ids=["below", "above"])
def test_contextmanager_body_exception(
    make: Callable[[list[str]], Callable[[], contextlib.AbstractContextManager[str]]],
) -> None:
    out: list[str] = []
    cm = make(out)
    with pytest.raises(ValueError, match="boom"), cm():
        raise ValueError("boom")
    assert out == ["enter", "deferred"]


def test_asynccontextmanager_raises() -> None:
    with pytest.raises(TypeError, match="async generator"):

        @defers_collector
        @contextlib.asynccontextmanager
        async def above() -> AsyncIterator[None]:
            yield

    with pytest.raises(TypeError, match="async generator"):

        @contextlib.asynccontextmanager
        @defers_collector
        async def below() -> AsyncIterator[None]:
            yield


################################################################################
# exceptions


def test_body_exception_propagates_and_defers_run() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(lambda: out.append("2"))
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        f()
    assert out == ["2", "1"]


def test_defers_after_exception_are_not_registered() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("before"))
        raise ValueError
        defer(lambda: out.append("after"))  # type: ignore[unreachable]

    with pytest.raises(ValueError):
        f()
    assert out == ["before"]


def test_failing_defer_does_not_stop_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out: list[str] = []

    def boom() -> None:
        raise RuntimeError("defer failed")

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(boom)
        defer(lambda: out.append("3"))

    f()
    assert out == ["3", "1"]
    err = caplog.text
    assert "Error in defer" in err
    assert "RuntimeError: defer failed" in err


def test_failing_defer_does_not_mask_body_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom() -> None:
        raise RuntimeError("defer failed")

    @defers_collector
    def f() -> None:
        defer(boom)
        raise ValueError("body failed")

    with pytest.raises(ValueError, match="body failed"):
        f()
    assert "RuntimeError: defer failed" in caplog.text


def test_failing_defer_reports_type_and_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom() -> None:
        raise ValueError

    @defers_collector
    def f() -> None:
        defer(boom)

    f()
    [record] = caplog.records
    assert record.name == "defer"
    assert record.levelno == logging.ERROR
    err = caplog.text
    assert "Traceback (most recent call last)" in err
    assert "in boom" in err
    assert err.rstrip().endswith("ValueError")


def test_defer_inside_deferred_call_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out: list[str] = []

    def first() -> None:
        out.append("first")
        defer(lambda: out.append("never"))

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("last"))
        defer(first)

    f()
    assert out == ["first", "last"]
    assert "must be called directly" in caplog.text


class _BrokenStr(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str for you")


class _BrokenStream:
    def write(self, s: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError


@pytest.mark.parametrize("stderr", [None, _BrokenStream()], ids=["none", "broken"])
def test_unusable_stderr_does_not_break_unwind(
    stderr: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    out: list[str] = []

    def boom() -> None:
        raise RuntimeError("defer failed")

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(boom)
        raise ValueError("body failed")

    # no handlers anywhere, so logging falls back to writing to sys.stderr
    monkeypatch.setattr(logging.getLogger("defer"), "propagate", False)
    monkeypatch.setattr(sys, "stderr", stderr)
    with pytest.raises(ValueError, match="body failed"):
        f()
    assert out == ["1"]


def test_exception_with_broken_str_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out: list[str] = []

    def boom() -> None:
        raise _BrokenStr

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(boom)

    f()
    assert out == ["1"]
    assert "_BrokenStr" in caplog.text


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_base_exceptions_in_defer_propagate_after_the_rest(
    exc: type[BaseException], caplog: pytest.LogCaptureFixture
) -> None:
    out: list[str] = []

    def raiser() -> None:
        raise exc("x")

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(raiser)
        defer(lambda: out.append("3"))

    with pytest.raises(exc, match="x"):
        f()
    assert out == ["3", "1"]
    assert caplog.text == ""


def test_base_exception_in_defer_replaces_body_exception() -> None:
    @defers_collector
    def f() -> None:
        defer(lambda: sys.exit(3))
        raise ValueError("body failed")

    with pytest.raises(SystemExit) as info:
        f()
    assert info.value.code == 3
    assert isinstance(info.value.__context__, ValueError)


class _Stop(BaseException):
    pass


@pytest.mark.parametrize("deferred", [_Stop, SystemExit, KeyboardInterrupt])
def test_in_flight_interrupt_is_not_replaced(
    deferred: type[BaseException], caplog: pytest.LogCaptureFixture
) -> None:
    original = _Stop("body")

    def raiser() -> None:
        raise deferred("defer")

    @defers_collector
    def f() -> None:
        defer(raiser)
        raise original

    with pytest.raises(_Stop) as info:
        f()
    assert info.value is original
    assert f"{deferred.__name__}: defer" in caplog.text


def test_in_flight_interrupt_is_not_replaced_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = _Stop("body")

    async def raiser() -> None:
        raise SystemExit("defer")

    @defers_collector
    async def f() -> None:
        defer(raiser)
        raise original

    with pytest.raises(_Stop) as info:
        asyncio.run(f())
    assert info.value is original
    assert "SystemExit: defer" in caplog.text


def test_only_first_base_exception_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    @defers_collector
    def f() -> None:
        defer(lambda: sys.exit(3))
        defer(interrupt)

    with pytest.raises(KeyboardInterrupt):
        f()
    assert "SystemExit: 3" in caplog.text


def test_non_callable_defer_raises_at_call_site() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(42)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be callable, not int"):
        f()
    assert out == ["1"]


################################################################################
# threads


def test_threads_have_separate_collectors() -> None:
    out: dict[int, list[int]] = {}
    barrier = threading.Barrier(4)

    @defers_collector
    def worker(n: int) -> None:
        out[n] = []
        for j in range(5):
            defer(lambda j=j: out[n].append(j))
        barrier.wait()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == {n: [4, 3, 2, 1, 0] for n in range(4)}


def test_defer_in_spawned_thread_raises() -> None:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            defer(lambda: None)
        except RuntimeError as e:
            errors.append(e)

    @defers_collector
    def f() -> None:
        t = threading.Thread(target=target)
        t.start()
        t.join()

    f()
    assert len(errors) == 1


################################################################################
# return values


def test_returns_value() -> None:
    @defers_collector
    def f() -> int:
        return 42

    assert f() == 42


def test_returns_value_with_defers() -> None:
    out: list[str] = []

    @defers_collector
    def f() -> str:
        defer(lambda: out.append("deferred"))
        return "value"

    assert f() == "value"
    assert out == ["deferred"]


def test_async() -> None:
    out: list[str] = []

    @defers_collector
    async def f() -> int:
        defer(lambda: out.append("deferred"))
        await asyncio.sleep(0)
        out.append("body")
        return 42

    assert asyncio.run(f()) == 42
    assert out == ["body", "deferred"]


def test_async_concurrent_tasks_have_separate_collectors() -> None:
    out: list[str] = []

    @defers_collector
    async def f(name: str) -> None:
        defer(lambda: out.append(f"{name} deferred"))
        await asyncio.sleep(0)
        out.append(f"{name} body")

    async def main() -> None:
        await asyncio.gather(f("a"), f("b"))

    asyncio.run(main())
    assert out == ["a body", "a deferred", "b body", "b deferred"]


def test_async_is_still_a_coroutine_function() -> None:
    @defers_collector
    async def f() -> None:
        pass

    assert inspect.iscoroutinefunction(f)


def test_async_deferred_callables_are_awaited() -> None:
    out: list[str] = []

    async def cleanup(name: str) -> None:
        await asyncio.sleep(0)
        out.append(name)

    @defers_collector
    async def f() -> None:
        defer(lambda: cleanup("async 1"))
        defer(lambda: out.append("sync"))
        defer(lambda: cleanup("async 2"))

    asyncio.run(f())
    assert out == ["async 2", "sync", "async 1"]


def test_failing_async_deferred_callable_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out: list[str] = []

    async def boom() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("async defer failed")

    @defers_collector
    async def f() -> None:
        defer(lambda: out.append("1"))
        defer(boom)

    asyncio.run(f())
    assert out == ["1"]
    assert "RuntimeError: async defer failed" in caplog.text


def test_async_deferred_callable_in_sync_function_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out: list[str] = []
    coros: list[Coroutine[Any, Any, None]] = []

    async def cleanup() -> None:
        out.append("never")

    def make() -> Coroutine[Any, Any, None]:
        coros.append(cleanup())
        return coros[-1]

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(make)

    f()
    assert out == ["1"]
    assert inspect.getcoroutinestate(coros[0]) == inspect.CORO_CLOSED
    assert "outside an async function" in caplog.text


def test_awaitable_deferred_in_sync_function_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    out: list[str] = []

    class Awaitable:
        def __await__(self) -> Generator[None, None, None]:
            yield

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(Awaitable)

    f()
    assert out == ["1"]
    assert "outside an async function" in caplog.text


def test_defer_in_spawned_task_raises() -> None:
    async def undecorated() -> None:
        defer(lambda: None)

    @defers_collector
    async def f() -> None:
        await asyncio.create_task(undecorated())

    with pytest.raises(RuntimeError):
        asyncio.run(f())


def test_cancelled_task_runs_defers() -> None:
    out: list[str] = []
    started = asyncio.Event()

    async def cleanup() -> None:
        await asyncio.sleep(0)
        out.append("async cleanup")

    @defers_collector
    async def f() -> None:
        defer(cleanup)
        defer(lambda: out.append("sync cleanup"))
        started.set()
        await asyncio.sleep(10)

    async def main() -> None:
        task = asyncio.create_task(f())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert out == ["sync cleanup", "async cleanup"]


def test_cancelled_error_in_deferred_callable_propagates() -> None:
    out: list[str] = []

    async def cancelled() -> None:
        raise asyncio.CancelledError

    @defers_collector
    async def f() -> None:
        defer(lambda: out.append("1"))
        defer(cancelled)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(f())
    assert out == ["1"]


################################################################################
# script


def test_main() -> None:
    res = subprocess.run(
        [sys.executable, str(HERE / "defer.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout == "Start\nEnd\nDefer called!\n"


EXAMPLE_OUTPUT = """\
basic: LIFO on return
  open a
  open b
  work
  close b
  close a
failing: deferred calls run on exceptions too
  open db
  close db
  caught ValueError('boom')
cleanup_fails: errors in deferred calls are logged, not raised
  work
  logged: Error in defer: ZeroDivisionError('division by zero')
  earlier cleanup still runs
acquire: hand-off
  open conn
  handed off, still open
  close conn
  open conn
  close conn
  caught ValueError('setup failed')
session: generators and contextmanager
  open session
  inside
  close session
many: comprehensions
  open f0
  open f1
  open f2
  close f2
  close f1
  close f0
fetch: async, deferred coroutines are awaited
  open sock
  fetched
  aclose sock
misuse: defer outside the decorated function's own body
  defer() must be called directly in a @defers_collector function, not in helper
"""


def test_example() -> None:
    res = subprocess.run(
        [sys.executable, str(HERE / "example.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout == EXAMPLE_OUTPUT
