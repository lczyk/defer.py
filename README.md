# defer.py

`defer` in python. go-like. single file, no dependencies, python 3.11+.

```python
from defer import defer, defers_collector

@defers_collector
def work() -> None:
    a = Resource("a")
    defer(a.close)
    b = Resource("b")
    defer(b.close)
    ...
    # b.close(), then a.close(), run when work() returns -- or raises
```

`defer.py` is one file with no dependencies, so just copy it into your project.

based on a [post](https://habr.com/en/articles/191786/) by Denis Kolodin.

## rules

- `defer(fn)` registers `fn` to run when the enclosing `@defers_collector` function
  exits, on return or on exception. last in, first out.
- `defer` must be called directly in the body of that function. comprehensions in
  the body are fine. anywhere else -- a helper, a thread, a task, a closure called
  later -- raises `RuntimeError`, instead of quietly attaching to whatever else is up
  the stack.
- `defer` returns its argument, so `@defer` on a nested `def` works for cleanups longer
  than a lambda (see [hand-off](#hand-off) below).
- keep `@defers_collector` closest to the `def`. a decorator written in python sitting
  between the two hides the body from `defer`.

## works with

- functions and methods, incl. `staticmethod` / `classmethod` in either order
- `async def`. deferred coroutines, e.g. `defer(conn.aclose)`, are awaited
- generators. deferred calls run when the generator finishes or is closed
- `@contextlib.contextmanager`, on either side of it

not supported: async generator functions, and sitting above `@property` (put it below).
both raise `TypeError` at decoration time.

## errors

- an exception from a deferred call is logged to the `defer` logger and the remaining
  deferred calls still run. with no logging configured it ends up on stderr.
- `KeyboardInterrupt`, `SystemExit` and other non-`Exception` errors from deferred calls
  propagate, once every deferred call has run. if the function is already propagating
  one, that one wins and the new one is logged.

## hand-off

no special api, just the go idiom -- a flag:

```python
@defers_collector
def acquire() -> Resource:
    res = Resource("conn")
    handed_off = False

    @defer
    def release() -> None:
        if not handed_off:
            res.close()

    setup(res)  # may raise, closing res on the way out
    handed_off = True
    return res
```

## vs `contextlib.ExitStack`

the stdlib's closest thing: a context manager holding a stack of cleanup callbacks.

what | `ExitStack` | `defer.py`
--- | --- | ---
scope | a `with` block | the whole decorated function, flat body
helpers | pass the stack around | not allowed
hand-off | `stack.pop_all()` | flag idiom
a callback raises | the rest still run, last exception wins, earlier ones chained only if the body raised, else lost | the rest still run, `Exception`s logged, first interrupt propagates
async | separate `AsyncExitStack` | same decorator, coroutines awaited
magic | none | frame inspection (`sys._getframe`)

reach for `ExitStack` if you want no magic, or need to pass cleanup between functions.
reach for `defer.py` for a flat body and the logging / interrupt policy.

## speed

`make bench`, python 3.14 on an m3 mac:

```
one cleanup per call, us per call
  no cleanup                      0.02
  try/finally                     0.02
  contextlib.ExitStack            0.68
  @defers_collector + defer       0.79

defer() cost by stack depth, us per defer
  depth     lexical   inspect.stack
      0        0.57          157.40
    100        0.59          811.45
    500        0.73         3454.00
```

about the same as `ExitStack`, and flat in stack depth. `inspect.stack` is the lookup
`defer` used before 0.2.0, kept in the benchmark for comparison.

## development

`make verify` runs ruff, mypy and the tests. `make tox` runs them on python 3.11 to 3.14
via [tox-uv](https://github.com/tox-dev/tox-uv), with the dev tools at their lowest allowed
versions. `make example` runs [example.py](example.py), `make bench` runs
[bench_defer.py](bench_defer.py).
