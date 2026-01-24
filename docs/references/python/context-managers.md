# Context Managers and the with Statement

This document explains how context managers work in Python—the mechanics behind the `with` statement, how to write custom context managers, and practical patterns for resource management.

By the end of this guide, we will understand when and why to use context managers, how to create them using classes and decorators, and how to handle async resources.

---

## 1. The Problem Context Managers Solve

Resources that need setup and cleanup are error-prone:

```python
# Manual resource management - error-prone
f = open("data.txt")
try:
    data = f.read()
finally:
    f.close()  # Must remember to close!
```

What if we forget the `try/finally`? What if we return early? What if an exception occurs? We risk resource leaks.

Context managers solve this by guaranteeing cleanup:

```python
# With context manager - cleanup is automatic
with open("data.txt") as f:
    data = f.read()
# f is automatically closed, no matter what
```

The `with` statement ensures cleanup runs even if an exception occurs inside the block.

---

## 2. What the with Statement Does

The `with` statement is syntactic sugar for a try/finally pattern.

### The Mechanics

```python
with expression as variable:
    # body
```

Is (approximately) equivalent to:

```python
manager = expression
variable = manager.__enter__()
try:
    # body
finally:
    manager.__exit__(exc_type, exc_val, exc_tb)
```

### The Protocol

A context manager is any object that implements:
- `__enter__(self)`: Called when entering the `with` block. Returns a value (often `self`).
- `__exit__(self, exc_type, exc_val, exc_tb)`: Called when exiting, with exception info if one occurred.

```python
class MyContextManager:
    def __enter__(self):
        print("Entering")
        return self  # This becomes the 'as' variable
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting")
        return False  # Don't suppress exceptions

with MyContextManager() as cm:
    print("Inside")
```

Output:
```
Entering
Inside
Exiting
```

---

## 3. Writing Context Managers as Classes

For full control, implement the protocol as a class.

### Basic Structure

```python
class Resource:
    def __init__(self, name):
        self.name = name
        self.acquired = False
    
    def __enter__(self):
        print(f"Acquiring {self.name}")
        self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Releasing {self.name}")
        self.acquired = False
        return False  # Don't suppress exceptions

with Resource("database") as r:
    print(f"Using {r.name}, acquired={r.acquired}")
```

Output:
```
Acquiring database
Using database, acquired=True
Releasing database
```

### The __exit__ Parameters

`__exit__` receives exception information:

```python
class Debugger:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False

with Debugger():
    raise ValueError("Something went wrong")
```

Output:
```
Exception occurred: ValueError: Something went wrong
Traceback (most recent call last):
  ...
ValueError: Something went wrong
```

### Suppressing Exceptions

If `__exit__` returns `True`, the exception is suppressed:

```python
class Suppress:
    def __init__(self, *exception_types):
        self.exceptions = exception_types
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            print(f"Suppressed {exc_type.__name__}")
            return True  # Suppress the exception
        return False

with Suppress(ValueError, TypeError):
    raise ValueError("Ignored!")

print("Execution continues")  # This runs because exception was suppressed
```

Use exception suppression carefully—silently swallowing errors makes debugging hard.

---

## 4. Using contextlib.contextmanager

Writing a class for simple context managers is verbose. The `contextlib.contextmanager` decorator is simpler.

### Basic Usage

```python
from contextlib import contextmanager

@contextmanager
def resource(name):
    print(f"Acquiring {name}")
    try:
        yield name  # Everything before yield is __enter__
    finally:
        print(f"Releasing {name}")  # This is __exit__

with resource("database") as r:
    print(f"Using {r}")
```

The function has three parts:
1. **Before yield**: Setup (runs in `__enter__`)
2. **yield**: Provides the value for `as` clause
3. **After yield**: Cleanup (runs in `__exit__`)

### The try/finally Is Important

Always use try/finally to ensure cleanup runs even if an exception occurs:

```python
@contextmanager
def bad_resource(name):
    print(f"Acquiring {name}")
    yield name
    print(f"Releasing {name}")  # Never runs if exception!

@contextmanager
def good_resource(name):
    print(f"Acquiring {name}")
    try:
        yield name
    finally:
        print(f"Releasing {name}")  # Always runs
```

### Yielding vs Returning

The `yield` statement provides the value that appears after `as`. It does not have to yield anything:

```python
@contextmanager
def timing():
    import time
    start = time.perf_counter()
    try:
        yield  # No value needed
    finally:
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.3f}s")

with timing():
    do_something()  # Elapsed: 0.123s
```

---

## 5. Practical Context Manager Patterns

### Pattern 1: Timer

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label=""):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s" if label else f"{elapsed:.4f}s")

with timer("Database query"):
    result = query_database()
```

### Pattern 2: Temporary Directory

```python
from contextlib import contextmanager
import tempfile
import shutil
import os

@contextmanager
def temp_directory():
    """Create a temporary directory that's cleaned up after use."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)

with temp_directory() as tmpdir:
    # Work with tmpdir
    file_path = os.path.join(tmpdir, "data.txt")
    with open(file_path, "w") as f:
        f.write("temporary data")
# Directory is automatically deleted
```

### Pattern 3: Change and Restore State

```python
from contextlib import contextmanager
import os

@contextmanager
def working_directory(path):
    """Temporarily change working directory."""
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)

with working_directory("/tmp"):
    # We're in /tmp here
    print(os.getcwd())
# Back to original directory
```

### Pattern 4: Database Transaction

```python
from contextlib import contextmanager

@contextmanager
def transaction(connection):
    """Commit on success, rollback on failure."""
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

with transaction(db_connection) as conn:
    conn.execute("INSERT INTO users (name) VALUES ('Alice')")
    conn.execute("INSERT INTO users (name) VALUES ('Bob')")
# Both committed, or both rolled back
```

### Pattern 5: Lock Acquisition

```python
import threading
from contextlib import contextmanager

lock = threading.Lock()

@contextmanager
def timed_lock(lock, timeout=5):
    """Acquire lock with timeout."""
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError("Could not acquire lock")
    try:
        yield
    finally:
        lock.release()

with timed_lock(lock, timeout=10):
    # We have the lock
    do_critical_work()
```

### Pattern 6: Redirect Output

```python
from contextlib import contextmanager, redirect_stdout
import io

@contextmanager
def capture_output():
    """Capture stdout to a string."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        yield buffer

with capture_output() as output:
    print("Hello!")
    print("World!")

captured = output.getvalue()
print(f"Captured: {captured!r}")  # 'Hello!\nWorld!\n'
```

---

## 6. Nesting Context Managers

Multiple context managers can be nested or combined.

### Nested with Statements

```python
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        outfile.write(infile.read())
```

### Multiple on One Line

```python
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())
```

### Parentheses for Readability (Python 3.10+)

```python
with (
    open("input.txt") as infile,
    open("output.txt", "w") as outfile,
    timer("Copy operation"),
):
    outfile.write(infile.read())
```

---

## 7. ExitStack: Dynamic Context Management

Sometimes we do not know at code time how many context managers we need.

### Basic ExitStack Usage

```python
from contextlib import ExitStack

with ExitStack() as stack:
    files = [stack.enter_context(open(f"file_{i}.txt")) for i in range(3)]
    # All files will be closed when exiting the with block
```

### Building Up Contexts

```python
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        # Open all files
        files = []
        for name in filenames:
            f = stack.enter_context(open(name))
            files.append(f)
        
        # Process all files
        for f in files:
            process(f)
    # All files closed here

process_files(["a.txt", "b.txt", "c.txt"])
```

### Cleanup Callbacks

```python
from contextlib import ExitStack

def cleanup():
    print("Cleanup called!")

with ExitStack() as stack:
    stack.callback(cleanup)  # Will be called on exit
    print("Doing work...")
# Output:
# Doing work...
# Cleanup called!
```

### Conditional Context Managers

```python
from contextlib import ExitStack, nullcontext

def process(use_lock=False, lock=None):
    with ExitStack() as stack:
        if use_lock:
            stack.enter_context(lock)
        
        do_work()

# Or using nullcontext (does nothing)
def process(use_lock=False, lock=None):
    ctx = lock if use_lock else nullcontext()
    with ctx:
        do_work()
```

---

## 8. Async Context Managers

For async code, use `async with` and the async protocol.

### The Async Protocol

```python
class AsyncResource:
    async def __aenter__(self):
        print("Async acquiring")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Async releasing")
        await asyncio.sleep(0.1)
        return False

async def main():
    async with AsyncResource() as r:
        print("Using async resource")

asyncio.run(main())
```

### Using asynccontextmanager

```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def async_timer(label=""):
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

async def main():
    async with async_timer("API call"):
        await asyncio.sleep(1)

asyncio.run(main())
```

### Practical Async Examples

```python
@asynccontextmanager
async def http_client():
    """Manage async HTTP client lifecycle."""
    import httpx
    client = httpx.AsyncClient()
    try:
        yield client
    finally:
        await client.aclose()

async def main():
    async with http_client() as client:
        response = await client.get("https://example.com")
        print(response.status_code)
```

```python
@asynccontextmanager
async def database_connection(url):
    """Manage async database connection."""
    import asyncpg
    conn = await asyncpg.connect(url)
    try:
        yield conn
    finally:
        await conn.close()

async def main():
    async with database_connection("postgresql://...") as conn:
        result = await conn.fetch("SELECT * FROM users")
```

---

## 9. Built-in Context Managers

Python provides many context managers in the standard library.

### File Objects

```python
with open("file.txt") as f:
    data = f.read()
```

### Threading Locks

```python
import threading

lock = threading.Lock()

with lock:
    # Critical section
    pass
```

### Decimal Context

```python
from decimal import Decimal, localcontext

with localcontext() as ctx:
    ctx.prec = 50  # High precision in this block
    result = Decimal(1) / Decimal(7)
```

### Suppress Exceptions

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("maybe_missing.txt")
# No error even if file doesn't exist
```

### Redirect stdout/stderr

```python
from contextlib import redirect_stdout
import io

f = io.StringIO()
with redirect_stdout(f):
    print("This goes to f")

output = f.getvalue()
```

### Change Directory (Python 3.11+)

```python
from contextlib import chdir

with chdir("/tmp"):
    # Working in /tmp
    pass
# Back to original
```

---

## 10. Context Managers vs try/finally

When should we use context managers instead of try/finally?

### Use Context Managers When

- Resource management is reusable across multiple places
- The pattern is "acquire-use-release"
- We want to abstract away cleanup details
- We are working with standard resources (files, locks, connections)

### Use try/finally When

- Cleanup is unique to one location
- The logic is too simple to warrant abstraction
- We need complex exception handling

### Example: When Context Manager Is Better

```python
# Repeated pattern - use context manager
class Database:
    @contextmanager
    def transaction(self):
        self.begin()
        try:
            yield
            self.commit()
        except:
            self.rollback()
            raise

# Now it's reusable
with db.transaction():
    db.execute(query1)
    db.execute(query2)
```

### Example: When try/finally Is Fine

```python
# One-off, simple cleanup
def process():
    temp_file = create_temp_file()
    try:
        do_work(temp_file)
    finally:
        os.remove(temp_file)
```

---

## 11. Common Mistakes

### Mistake 1: Forgetting try/finally in @contextmanager

```python
# BAD - cleanup might not run
@contextmanager
def resource():
    acquire()
    yield
    release()  # Skipped if exception!

# GOOD
@contextmanager
def resource():
    acquire()
    try:
        yield
    finally:
        release()
```

### Mistake 2: Returning Instead of Yielding

```python
# BAD - not a context manager!
@contextmanager
def resource():
    return "value"  # This raises StopIteration

# GOOD
@contextmanager
def resource():
    yield "value"
```

### Mistake 3: Using the Value Outside the Block

```python
with open("file.txt") as f:
    data = f.read()

# f is closed here!
print(f.read())  # ValueError: I/O operation on closed file
```

### Mistake 4: Ignoring the Return Value of __exit__

```python
# If __exit__ returns True, exception is suppressed
# Only do this intentionally!

class Risky:
    def __exit__(self, *args):
        return True  # All exceptions suppressed!
```

---

## 12. Context Managers in Testing

Context managers are useful for test fixtures.

### Mock Context Manager

```python
from contextlib import contextmanager
from unittest.mock import patch

@contextmanager
def mock_api():
    with patch("mymodule.api_call") as mock:
        mock.return_value = {"status": "ok"}
        yield mock

def test_something():
    with mock_api():
        result = function_that_calls_api()
        assert result["status"] == "ok"
```

### Temporary State

```python
@contextmanager
def env_var(name, value):
    """Temporarily set environment variable."""
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            del os.environ[name]
        else:
            os.environ[name] = old_value

def test_with_env():
    with env_var("DATABASE_URL", "test://localhost"):
        # Test code here
        pass
```

---

## Summary

Context managers provide automatic resource management through the `with` statement. They guarantee cleanup runs even when exceptions occur.

Key points:
- `__enter__` sets up resources, `__exit__` cleans up
- `@contextmanager` decorator simplifies creating context managers
- Always use try/finally in `@contextmanager` functions
- `ExitStack` handles dynamic numbers of context managers
- `async with` works with `__aenter__`/`__aexit__`

When to use context managers:
- File handling
- Database connections and transactions
- Locks and synchronization
- Temporary state changes
- Timing and profiling
- Any acquire-use-release pattern

For LLM applications, context managers are useful for:
- Managing API client sessions
- Database transactions for storing embeddings
- Acquiring rate-limiting semaphores
- Timing inference operations
- Temporary file handling for document processing

The mental model: a context manager wraps a block of code with guaranteed setup and teardown, making resource leaks nearly impossible when used correctly.
