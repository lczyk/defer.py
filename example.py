"""Examples of defer.py usage. Run with `python example.py`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator

from defer import defer, defers_collector


class Resource:
    def __init__(self, name: str) -> None:
        self.name = name
        print(f"  open {name}")

    def close(self) -> None:
        print(f"  close {self.name}")

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        print(f"  aclose {self.name}")


@defers_collector
def basic() -> None:
    a = Resource("a")
    defer(a.close)
    b = Resource("b")
    defer(b.close)
    print("  work")


@defers_collector
def failing() -> None:
    db = Resource("db")
    defer(db.close)
    raise ValueError("boom")


@defers_collector
def cleanup_fails() -> None:
    defer(lambda: print("  earlier cleanup still runs"))
    defer(lambda: 1 / 0)
    print("  work")


@defers_collector
def acquire(fail: bool) -> Resource:
    # go idiom: close on the way out, unless ownership was handed to the caller
    res = Resource("conn")
    handed_off = False

    @defer
    def release() -> None:
        if not handed_off:
            res.close()

    if fail:
        raise ValueError("setup failed")
    handed_off = True
    return res


@contextlib.contextmanager
@defers_collector
def session() -> Iterator[Resource]:
    s = Resource("session")
    defer(s.close)
    yield s


@defers_collector
def many() -> None:
    files = [Resource(f"f{i}") for i in range(3)]
    [defer(f.close) for f in files]


@defers_collector
async def fetch() -> None:
    sock = Resource("sock")
    defer(sock.aclose)
    await asyncio.sleep(0)
    print("  fetched")


def helper() -> None:
    defer(lambda: None)


@defers_collector
def misuse() -> None:
    helper()


class Logged(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        exc = record.exc_info[1] if record.exc_info else None
        print(f"  logged: {record.getMessage()}: {exc!r}")


def main() -> None:
    logging.getLogger("defer").addHandler(Logged())

    print("basic: LIFO on return")
    basic()

    print("failing: deferred calls run on exceptions too")
    try:
        failing()
    except ValueError as e:
        print(f"  caught {e!r}")

    print("cleanup_fails: errors in deferred calls are logged, not raised")
    cleanup_fails()

    print("acquire: hand-off")
    conn = acquire(fail=False)
    print("  handed off, still open")
    conn.close()
    try:
        acquire(fail=True)
    except ValueError as e:
        print(f"  caught {e!r}")

    print("session: generators and contextmanager")
    with session():
        print("  inside")

    print("many: comprehensions")
    many()

    print("fetch: async, deferred coroutines are awaited")
    asyncio.run(fetch())

    print("misuse: defer outside the decorated function's own body")
    try:
        misuse()
    except RuntimeError as e:
        print(f"  {e}")


if __name__ == "__main__":
    main()
