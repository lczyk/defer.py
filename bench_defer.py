"""Micro-benchmarks for defer.py. Run with `python bench_defer.py`."""

import contextlib
import inspect
import sys
import timeit
from collections.abc import Callable

from defer import defer, defers_collector


def per_op_us(fn: Callable[[], object], ops: int = 1) -> float:
    """Best-of-5 time per call of ``fn``, divided by the ``ops`` it performs."""
    timer = timeit.Timer(fn)
    number = 1
    while timer.timeit(number) < 0.05:
        number *= 2
    return min(timer.repeat(repeat=5, number=number)) / number / ops * 1e6


def at_depth(depth: int, fn: Callable[[], object]) -> object:
    return fn() if depth == 0 else at_depth(depth - 1, fn)


def noop() -> None:
    pass


def plain() -> None:
    noop()


def try_finally() -> None:
    try:
        pass
    finally:
        noop()


def exit_stack() -> None:
    with contextlib.ExitStack() as stack:
        stack.callback(noop)


@defers_collector
def with_defer() -> None:
    defer(noop)


def stack_walk_defer(x: Callable[[], None]) -> None:
    # the lookup defer() used before lexical scoping, for comparison
    for f in inspect.stack():
        if "__defers__" in f[0].f_locals:
            f[0].f_locals["__defers__"].append(x)
            break


LEXICAL_PER_CALL = 1000
WALK_PER_CALL = 5


@defers_collector
def lexical_defers() -> None:
    for _ in range(LEXICAL_PER_CALL):
        defer(noop)


@defers_collector
def stack_walk_defers() -> None:
    for _ in range(WALK_PER_CALL):
        stack_walk_defer(noop)


def main() -> None:
    print(f"python {sys.version.split()[0]}")

    print("\none cleanup per call, us per call")
    for name, fn in [
        ("no cleanup", plain),
        ("try/finally", try_finally),
        ("contextlib.ExitStack", exit_stack),
        ("@defers_collector + defer", with_defer),
    ]:
        print(f"  {name:<28}{per_op_us(fn):8.2f}")

    print("\ndefer() cost by stack depth, us per defer")
    print(f"  {'depth':>5}{'lexical':>12}{'inspect.stack':>16}")
    for depth in (0, 100, 500):
        lexical = per_op_us(lambda: at_depth(depth, lexical_defers), LEXICAL_PER_CALL)
        walk = per_op_us(lambda: at_depth(depth, stack_walk_defers), WALK_PER_CALL)
        print(f"  {depth:>5}{lexical:>12.2f}{walk:>16.2f}")


if __name__ == "__main__":
    main()
