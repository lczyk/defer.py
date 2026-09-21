import subprocess
import sys
import threading
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
    assert "Error in defer: defer failed" in capsys.readouterr().err


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
    assert "Error in defer: defer failed" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_base_exceptions_in_defer_are_swallowed(
    exc: type[BaseException], capsys: pytest.CaptureFixture[str]
) -> None:
    out: list[str] = []

    def raiser() -> None:
        raise exc("x")

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(raiser)

    f()
    assert out == ["1"]
    assert "Error in defer: x" in capsys.readouterr().err


def test_non_callable_defer_reports_error(capsys: pytest.CaptureFixture[str]) -> None:
    out: list[str] = []

    @defers_collector
    def f() -> None:
        defer(lambda: out.append("1"))
        defer(42)  # type: ignore[arg-type]

    f()
    assert out == ["1"]
    assert "Error in defer:" in capsys.readouterr().err


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
# known issues


@pytest.mark.xfail(strict=True, reason="wrapped discards func's return value")
def test_returns_value() -> None:
    @defers_collector
    def f() -> int:
        return 42

    assert f() == 42


@pytest.mark.xfail(
    strict=True, reason="coroutine is dropped with the return value, never awaited"
)
@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
def test_async() -> None:
    import asyncio

    out: list[str] = []

    @defers_collector
    async def f() -> None:
        defer(lambda: out.append("deferred"))
        out.append("body")

    asyncio.run(f())  # type: ignore[arg-type]
    assert out == ["body", "deferred"]


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
