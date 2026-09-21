import asyncio
import inspect
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from pathlib import Path

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


def test_defer_in_nested_non_collector_function() -> None:
    # defer attaches to the nearest collector up the stack, not the direct caller
    out: list[str] = []

    def helper() -> None:
        defer(lambda: out.append("helper"))
        out.append("helper body")

    @defers_collector
    def f() -> None:
        helper()
        out.append("f body")

    f()
    assert out == ["helper body", "f body", "helper"]


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


def test_no_defers() -> None:
    called = []

    @defers_collector
    def f() -> None:
        called.append(True)

    f()
    assert called == [True]


def test_defer_outside_collector_is_noop() -> None:
    out: list[str] = []
    defer(lambda: out.append("never"))
    assert out == []


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
    capsys: pytest.CaptureFixture[str],
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
    err = capsys.readouterr().err
    assert "Error in defer:" in err
    assert "RuntimeError: defer failed" in err


def test_failing_defer_does_not_mask_body_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> None:
        raise RuntimeError("defer failed")

    @defers_collector
    def f() -> None:
        defer(boom)
        raise ValueError("body failed")

    with pytest.raises(ValueError, match="body failed"):
        f()
    assert "RuntimeError: defer failed" in capsys.readouterr().err


def test_failing_defer_reports_type_and_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> None:
        raise ValueError

    @defers_collector
    def f() -> None:
        defer(boom)

    f()
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert "in boom" in err
    assert err.rstrip().endswith("ValueError")


def test_defer_during_unwind_runs() -> None:
    out: list[str] = []

    def first() -> None:
        out.append("first")
        defer(lambda: out.append("registered by first"))

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("last"))
        defer(first)

    f()
    assert out == ["first", "registered by first", "last"]


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

    monkeypatch.setattr(sys, "stderr", stderr)
    with pytest.raises(ValueError, match="body failed"):
        f()
    assert out == ["1"]


def test_exception_with_broken_str_is_reported(
    capsys: pytest.CaptureFixture[str],
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
    assert "_BrokenStr" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_base_exceptions_in_defer_propagate_after_the_rest(
    exc: type[BaseException], capsys: pytest.CaptureFixture[str]
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
    assert capsys.readouterr().err == ""


def test_base_exception_in_defer_replaces_body_exception() -> None:
    @defers_collector
    def f() -> None:
        defer(lambda: sys.exit(3))
        raise ValueError("body failed")

    with pytest.raises(SystemExit) as info:
        f()
    assert info.value.code == 3
    assert isinstance(info.value.__context__, ValueError)


def test_only_first_base_exception_propagates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    @defers_collector
    def f() -> None:
        defer(lambda: sys.exit(3))
        defer(interrupt)

    with pytest.raises(KeyboardInterrupt):
        f()
    assert "SystemExit: 3" in capsys.readouterr().err


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


def test_thread_does_not_see_callers_collector() -> None:
    # the stack walk is per-thread, so a defer in a spawned thread with no
    # collector of its own is dropped rather than attached to the spawner
    out: list[str] = []

    @defers_collector
    def f() -> None:
        t = threading.Thread(target=lambda: defer(lambda: out.append("thread")))
        t.start()
        t.join()

    f()
    assert out == []


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
