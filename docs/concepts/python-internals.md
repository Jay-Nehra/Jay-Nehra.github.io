---
tags:
  - python
  - concepts
  - interview
---

# Python Internals

Senior-level Python questions. Not "how do we write a list comprehension" but "what's actually happening under the hood."

---

## What is the GIL and when does it actually matter?

**What they're really asking**: Do we understand Python's concurrency model and its limitations?

**The core insight**: The Global Interpreter Lock (GIL) ensures only one thread executes Python bytecode at a time. But this matters less than people think for most AI workloads.

**When the GIL doesn't matter**:
- I/O-bound work (API calls, database queries, file reads) — threads release GIL while waiting
- C extensions (NumPy, pandas, model inference) — release GIL during computation
- `asyncio` — single-threaded anyway, GIL irrelevant

**When the GIL matters**:
- Pure Python CPU-bound work across threads
- Rarely our bottleneck in LLM applications

**The workarounds**:
- `multiprocessing` for CPU parallelism (separate processes, no shared GIL)
- `asyncio` for I/O concurrency (no threads needed)
- Offload to C/Rust extensions that release GIL

**Key point**: For LLM API calls, we're I/O-bound. The GIL isn't our problem—network latency is.

---

## How does Python's memory management work?

**What they're really asking**: Can we debug memory issues and understand object lifecycle?

**The two-layer system**:

1. **Reference counting** (primary): Every object tracks how many references point to it. When count hits zero, memory is freed immediately.

2. **Garbage collector** (backup): Handles circular references that reference counting can't detect.

```python
a = [1, 2, 3]  # refcount = 1
b = a          # refcount = 2
del a          # refcount = 1
del b          # refcount = 0 → freed immediately
```

**Circular reference problem**:
```python
a = []
b = []
a.append(b)
b.append(a)
del a, b  # refcounts are 1, not 0 — GC must clean up
```

**Practical implications**:
- Large objects freed immediately when dereferenced (good for memory)
- Circular references delay cleanup until GC runs
- `__del__` methods run at unpredictable times with cycles

**For AI workloads**: Watch out for caching large embeddings or model outputs. If we hold references, memory grows.

---

## Generators vs iterators — what's the difference and when do we use generators?

**What they're really asking**: Do we understand lazy evaluation and memory efficiency?

**The distinction**:
- **Iterator**: Any object with `__iter__` and `__next__` methods
- **Generator**: A function that uses `yield`, automatically creates an iterator

**The key insight**: Generators produce values one at a time, on demand. We don't store everything in memory.

```python
# Eager — stores all 1M items in memory
data = [process(x) for x in range(1_000_000)]

# Lazy — stores one item at a time
data = (process(x) for x in range(1_000_000))
```

**When to use generators**:
- Processing large files line by line
- Streaming API responses
- Pipeline of transformations on large data
- Anywhere memory matters more than random access

**When NOT to use generators**:
- Need to iterate multiple times (generator exhausts)
- Need length or indexing
- Data fits comfortably in memory

**For AI workloads**: Streaming LLM responses, processing embeddings in batches, chunking large documents.

---

## What happens when we `import` a module?

**What they're really asking**: Do we understand Python's module system and potential side effects?

**The sequence**:

1. **Check cache**: Is module already in `sys.modules`? If yes, return cached version.

2. **Find the module**: Search `sys.path` for the module file.

3. **Create module object**: Empty module object created.

4. **Execute module code**: The entire module file runs, top to bottom.

5. **Cache it**: Module stored in `sys.modules`.

**Critical implications**:

- **Top-level code runs on import**: 
  ```python
  # config.py
  print("Loading config...")  # Runs when imported!
  DB_URL = os.environ["DATABASE_URL"]  # Runs on import!
  ```

- **Imports are cached**: Second `import` returns the same object, doesn't re-execute.

- **Circular imports fail**: If A imports B and B imports A, one gets a partial module.

**Best practice**: Keep top-level code minimal. Move expensive operations inside functions.

---

## How do decorators work under the hood?

**What they're really asking**: Do we understand Python's first-class functions and the decorator pattern?

**The simple truth**: Decorators are just functions that take a function and return a function.

```python
@decorator
def func():
    pass

# Is exactly equivalent to:
def func():
    pass
func = decorator(func)
```

**A decorator with arguments**:
```python
@decorator(arg)
def func():
    pass

# Is equivalent to:
def func():
    pass
func = decorator(arg)(func)  # decorator(arg) returns the actual decorator
```

**The standard pattern**:
```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserves func's name, docstring
    def wrapper(*args, **kwargs):
        # Before
        result = func(*args, **kwargs)
        # After
        return result
    return wrapper
```

**Why `@wraps` matters**: Without it, the decorated function loses its `__name__`, `__doc__`, making debugging harder.

**Common uses in AI**: Retry logic, caching, timing, authentication, input validation.

---

## What's `async`/`await` actually doing?

**What they're really asking**: Do we understand cooperative concurrency vs parallelism?

**The mental model**: `async`/`await` is about *waiting efficiently*, not about *doing multiple things simultaneously*.

**What happens**:

1. `async def` creates a coroutine (a pausable function)
2. `await` pauses the coroutine and yields control to the event loop
3. Event loop runs other coroutines while we wait
4. When our I/O completes, event loop resumes us

```python
async def fetch_all():
    # These run concurrently — all start, then all wait together
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
```

**Key insight**: Only ONE coroutine runs at a time. Concurrency comes from switching during waits, not parallel execution.

**When it helps**:
- Many I/O operations (API calls, database queries)
- Network-bound work where we spend time waiting

**When it doesn't help**:
- CPU-bound work (use multiprocessing)
- Single sequential operation

**For LLM applications**: Perfect for concurrent API calls to multiple models or parallel embedding generation.

---

## Why are default mutable arguments dangerous?

**What they're really asking**: Do we understand Python's evaluation model?

**The trap**:
```python
def add_item(item, items=[]):
    items.append(item)
    return items

add_item("a")  # ['a']
add_item("b")  # ['a', 'b'] — Wait, what?!
```

**Why it happens**: Default arguments are evaluated once, at function definition time, not at call time. The same list object is reused across all calls.

**The fix**:
```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

**The rule**: Never use mutable defaults (`[]`, `{}`, `set()`). Use `None` and create inside the function.

**Where we see this in practice**: Function that builds up a list of messages, accumulates embeddings, or collects results.

---

## How does `__slots__` save memory?

**What they're really asking**: Do we understand Python's object model and when to optimize?

**Normal objects**: Each instance has a `__dict__` — a dictionary storing all attributes. Dictionaries have overhead (~100+ bytes).

**With `__slots__`**: No `__dict__`. Attributes stored in a fixed-size array. Much smaller.

```python
class Normal:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Slotted:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Slotted uses ~40-50% less memory per instance
```

**When to use**:
- Creating millions of small objects
- Memory-constrained environments
- Data classes with fixed attributes

**Tradeoffs**:
- Can't add arbitrary attributes
- Slightly complicates inheritance
- Premature optimization if we don't have many instances

**For AI workloads**: Rarely needed. Useful if we're storing millions of document chunks or embeddings as objects.

---

## What are descriptors and where do we see them?

**What they're really asking**: Do we understand Python's attribute access machinery?

**The definition**: A descriptor is any object that implements `__get__`, `__set__`, or `__delete__`.

**Where we see them without realizing**:
- `@property` — uses descriptors
- `@classmethod`, `@staticmethod` — descriptors
- Functions in classes — method binding is a descriptor
- ORMs (SQLAlchemy columns) — descriptors

**How `@property` works**:
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def area(self):
        return 3.14 * self._radius ** 2

# When we access c.area:
# 1. Python sees 'area' is a descriptor (has __get__)
# 2. Calls area.__get__(c, Circle)
# 3. Which calls the wrapped function
```

**Practical use**: We rarely write descriptors directly, but understanding them explains property behavior, ORM magic, and framework patterns.

---

## How does method resolution order (MRO) work?

**What they're really asking**: Do we understand multiple inheritance and super()?

**The problem**: With multiple inheritance, which parent's method gets called?

```python
class A:
    def method(self): print("A")

class B(A):
    def method(self): print("B")

class C(A):
    def method(self): print("C")

class D(B, C):
    pass

D().method()  # Prints "B" — but why?
```

**The C3 linearization algorithm** determines order. We can see it:
```python
D.__mro__  # (D, B, C, A, object)
```

**How `super()` uses MRO**:
```python
class B(A):
    def method(self):
        super().method()  # Calls next in MRO, not necessarily A!
```

**Practical rule**: Keep inheritance simple. Prefer composition. If using multiple inheritance, understand that `super()` follows MRO, not the explicit parent.

**In frameworks**: Mixin patterns rely on MRO. Understanding it helps debug unexpected method calls.

---

## What's the difference between `is` and `==`?

**What they're really asking**: Do we understand identity vs equality?

**The distinction**:
- `==` compares values (calls `__eq__`)
- `is` compares identity (same object in memory)

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b  # True — same value
a is b  # False — different objects
a is c  # True — same object
```

**The `None` idiom**: Always use `is None`, not `== None`.
- `is None` checks identity (fast, unambiguous)
- `== None` calls `__eq__` (could be overridden, slower)

**Integer caching gotcha**:
```python
a = 256
b = 256
a is b  # True — Python caches small integers

a = 257
b = 257
a is b  # False — not cached (usually)
```

**Rule**: Use `is` only for `None`, `True`, `False`, or when we explicitly need identity comparison.
