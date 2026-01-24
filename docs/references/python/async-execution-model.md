# Async/Await Execution Model

This document explains how Python's async/await actually works—not just the syntax, but the execution model underneath. We will understand what the event loop is, what coroutines are, and how concurrent execution happens with only one thread.

By the end of this guide, we will know when async helps, when it hurts, and how to write correct async code for I/O-bound workloads like API calls and database queries.

---

## 1. The Fundamental Concept

Before we dive into syntax, we need to understand what problem async/await solves and how it differs from threading.

### The Waiting Problem

Most programs spend a lot of time waiting. When we make an HTTP request, we wait for the network. When we query a database, we wait for the disk. When we read a file, we wait for the operating system.

During this waiting, our program could be doing something else. But in synchronous code, it just sits there:

```python
import requests

def fetch_all():
    # Each request waits for the previous one
    r1 = requests.get("https://api.example.com/user/1")  # Wait 100ms
    r2 = requests.get("https://api.example.com/user/2")  # Wait 100ms
    r3 = requests.get("https://api.example.com/user/3")  # Wait 100ms
    return [r1.json(), r2.json(), r3.json()]

# Total time: ~300ms (sequential waiting)
```

With threading, we can wait on all three simultaneously:

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_user(user_id):
    return requests.get(f"https://api.example.com/user/{user_id}").json()

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_user, [1, 2, 3]))

# Total time: ~100ms (parallel waiting)
```

Async/await is another solution to the same problem, but with a fundamentally different approach.

### The Key Insight

Async/await achieves concurrency with a single thread by cooperatively yielding control during waits. Instead of creating multiple threads that the operating system switches between, we have one thread that switches between tasks at explicit points.

This is like having one person juggle three phone calls: put caller 1 on hold, talk to caller 2, put caller 2 on hold, check if caller 1 is ready, etc. One person, multiple conversations.

### Single-Threaded Concurrency

This sounds like a contradiction. How can one thread do multiple things?

The answer: it does not. One thread does one thing at a time. But when that thing is "waiting for a network response," the thread can do something else instead of just waiting.

```python
import asyncio
import httpx

async def fetch_all():
    async with httpx.AsyncClient() as client:
        # All three requests are sent, then we wait for all of them
        r1 = await client.get("https://api.example.com/user/1")
        # While we await, other tasks could run
```

Wait, that still looks sequential. Let us do it concurrently:

```python
async def fetch_all():
    async with httpx.AsyncClient() as client:
        # Create all tasks at once
        tasks = [
            client.get("https://api.example.com/user/1"),
            client.get("https://api.example.com/user/2"),
            client.get("https://api.example.com/user/3"),
        ]
        # Wait for all of them concurrently
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# Total time: ~100ms (concurrent waiting)
```

---

## 2. The Event Loop

The event loop is the heart of async Python. It is a loop that:
1. Waits for events (I/O completions, timers, etc.)
2. Runs the callbacks associated with those events
3. Repeats forever

### What the Event Loop Does

Think of the event loop as a task scheduler. It maintains a list of tasks and decides which one to run next.

```
Event Loop:
┌─────────────────────────────────────────────┐
│  1. Check: Any tasks ready to run?          │
│  2. Pick one and run it until it awaits     │
│  3. Check: Any I/O operations completed?    │
│  4. Mark those tasks as ready               │
│  5. Go to step 1                            │
└─────────────────────────────────────────────┘
```

A task runs until it hits an `await`, at which point it voluntarily gives control back to the event loop. The event loop then picks another ready task to run.

### Running the Event Loop

In Python 3.7+, the simplest way to run async code:

```python
import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# Start the event loop and run main() to completion
asyncio.run(main())
```

`asyncio.run()` creates an event loop, runs our coroutine, and cleans up. It is the entry point from sync to async world.

What if we are already inside async code and want to call another async function? We just `await` it:

```python
async def helper():
    await asyncio.sleep(1)
    return 42

async def main():
    result = await helper()  # Call async function from async function
    print(result)

asyncio.run(main())
```

### The Event Loop Is Not Magic

The event loop is just Python code running in our thread. When we call `asyncio.run()`, we are essentially running this:

```python
# Simplified conceptual model
def run(coro):
    loop = create_event_loop()
    task = loop.create_task(coro)
    
    while not task.done():
        # Run one step of a ready task
        ready_task = loop.get_ready_task()
        if ready_task:
            ready_task.run_one_step()
        
        # Check for completed I/O
        completed_io = loop.check_io(timeout=0.1)
        for callback in completed_io:
            callback.task.mark_ready()
    
    return task.result()
```

This is why we must not "block the event loop"—if our code does something slow without yielding, no other tasks can run.

---

## 3. Coroutines: The Building Block

A coroutine is a function that can be paused and resumed. When we write `async def`, we are creating a coroutine function.

### Creating Coroutines

```python
async def greet(name):
    return f"Hello, {name}"

# Calling the function returns a coroutine object, not the result
coro = greet("Alice")
print(coro)  # <coroutine object greet at 0x...>

# To actually run it, we need to await it (or use asyncio.run)
asyncio.run(greet("Alice"))  # "Hello, Alice"
```

A coroutine object is like a paused function execution. It holds the function's state but has not started running yet.

### The Await Keyword

`await` does two things:
1. Starts or resumes a coroutine
2. Yields control to the event loop if the coroutine is not ready

```python
async def main():
    # await starts the coroutine and waits for its result
    result = await some_coroutine()
    
    # If some_coroutine needs to wait (e.g., for I/O),
    # the event loop can run other tasks during that wait
```

We can only use `await` inside an `async def` function. This is Python enforcing that async "infection" is explicit.

### What Happens at an Await

Let us trace what happens:

```python
async def fetch():
    print("Starting fetch")
    await asyncio.sleep(1)  # <-- What happens here?
    print("Fetch complete")
    return "data"

async def main():
    result = await fetch()
    print(result)

asyncio.run(main())
```

1. `asyncio.run(main())` starts the event loop and schedules `main()`
2. `main()` runs until it hits `await fetch()`
3. `fetch()` starts, prints "Starting fetch"
4. `fetch()` hits `await asyncio.sleep(1)`
5. `asyncio.sleep(1)` tells the event loop "wake me up in 1 second"
6. Control returns to the event loop
7. The event loop has nothing else to do, so it waits
8. After 1 second, the event loop wakes up `fetch()`
9. `fetch()` continues, prints "Fetch complete", returns "data"
10. `main()` continues with `result = "data"`

The key: `await` is a suspension point. The coroutine pauses there, and the event loop can run other things.

---

## 4. Running Tasks Concurrently

The power of async comes from running multiple tasks concurrently. There are several ways to do this.

### asyncio.gather: Wait for Multiple Tasks

`gather` runs multiple coroutines concurrently and collects their results:

```python
import asyncio

async def fetch(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network delay
    print(f"Done {url}")
    return f"Result from {url}"

async def main():
    # All three run concurrently
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
    print(results)

asyncio.run(main())
```

Output:
```
Fetching url1
Fetching url2
Fetching url3
Done url1
Done url2
Done url3
['Result from url1', 'Result from url2', 'Result from url3']
```

Notice: all three "Fetching" messages appear immediately, then all three "Done" messages appear together after 1 second. The total time is ~1 second, not 3.

`gather` returns results in the same order as the input coroutines, regardless of completion order.

### asyncio.create_task: Fire and Forget (Sort Of)

`create_task` schedules a coroutine to run, returning a `Task` object immediately:

```python
async def background_work():
    await asyncio.sleep(1)
    print("Background work done")

async def main():
    # Start the task but don't wait for it yet
    task = asyncio.create_task(background_work())
    
    print("Main continues immediately")
    await asyncio.sleep(0.5)
    print("Main did some other work")
    
    # Now wait for the background task
    await task
    print("All done")

asyncio.run(main())
```

Output:
```
Main continues immediately
Main did some other work
Background work done
All done
```

`create_task` is useful when we want to start work now but await it later.

### The Difference Between gather and create_task

```python
# These are equivalent:
results = await asyncio.gather(coro1(), coro2(), coro3())

# And:
task1 = asyncio.create_task(coro1())
task2 = asyncio.create_task(coro2())
task3 = asyncio.create_task(coro3())
results = [await task1, await task2, await task3]
```

`gather` is a convenience for the common case of running coroutines concurrently and waiting for all of them.

### asyncio.as_completed: Process Results as They Arrive

Sometimes we want to process results as soon as they are ready, not wait for all of them:

```python
import asyncio
import random

async def fetch(url):
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)
    return f"Result from {url} (took {delay:.1f}s)"

async def main():
    coroutines = [fetch(f"url{i}") for i in range(5)]
    
    for coro in asyncio.as_completed(coroutines):
        result = await coro
        print(result)  # Prints results as they complete

asyncio.run(main())
```

Results appear in completion order, not input order.

### asyncio.wait: More Control

`wait` gives us more control over how we wait for tasks:

```python
import asyncio

async def main():
    tasks = [
        asyncio.create_task(asyncio.sleep(1)),
        asyncio.create_task(asyncio.sleep(2)),
        asyncio.create_task(asyncio.sleep(3)),
    ]
    
    # Wait for the first one to complete
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    print(f"Done: {len(done)}, Pending: {len(pending)}")
    # Done: 1, Pending: 2

asyncio.run(main())
```

Options for `return_when`:
- `FIRST_COMPLETED`: Return when any task finishes
- `FIRST_EXCEPTION`: Return when any task raises
- `ALL_COMPLETED`: Return when all tasks finish (default)

---

## 5. Timeouts and Cancellation

Real applications need to handle slow operations gracefully.

### Timeouts with wait_for

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "done"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
        print(result)
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(main())  # Prints "Operation timed out!" after 2 seconds
```

`wait_for` cancels the underlying task when the timeout is reached.

### Timeout Context Manager (Python 3.11+)

```python
import asyncio

async def main():
    try:
        async with asyncio.timeout(2.0):
            await slow_operation()
    except TimeoutError:
        print("Timed out!")
```

### Manual Cancellation

```python
import asyncio

async def long_running():
    try:
        while True:
            print("Working...")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Cancelled! Cleaning up...")
        raise  # Always re-raise CancelledError

async def main():
    task = asyncio.create_task(long_running())
    
    await asyncio.sleep(3)
    task.cancel()  # Request cancellation
    
    try:
        await task  # Wait for cancellation to complete
    except asyncio.CancelledError:
        print("Task was cancelled")

asyncio.run(main())
```

`CancelledError` is raised at the next `await` point inside the cancelled task. We should catch it, do cleanup, and re-raise it.

---

## 6. Handling Errors

Errors in async code need careful handling, especially with concurrent tasks.

### Errors in gather

By default, if one task in `gather` raises, the exception propagates and other tasks continue running:

```python
import asyncio

async def good():
    await asyncio.sleep(1)
    return "good"

async def bad():
    await asyncio.sleep(0.5)
    raise ValueError("Something went wrong")

async def main():
    try:
        results = await asyncio.gather(good(), bad())
    except ValueError as e:
        print(f"Error: {e}")
        # The 'good' task is still running in the background!

asyncio.run(main())
```

To get all results and exceptions without raising immediately:

```python
async def main():
    results = await asyncio.gather(
        good(), bad(), good(),
        return_exceptions=True  # Return exceptions as values
    )
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Success: {result}")

asyncio.run(main())
```

Output:
```
Success: good
Error: Something went wrong
Success: good
```

### Task Groups (Python 3.11+)

Task groups provide better error handling—if one task fails, all others are cancelled:

```python
import asyncio

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(good())
            task2 = tg.create_task(bad())
            task3 = tg.create_task(good())
    except* ValueError as eg:
        print(f"Caught: {eg.exceptions}")
```

Task groups use exception groups (the `except*` syntax) to collect all exceptions.

---

## 7. The Blocking Problem

The most common async mistake is blocking the event loop. If our code does something slow without yielding, nothing else can run.

### What Blocks the Event Loop

```python
import asyncio
import time

async def blocking():
    print("Starting blocking operation")
    time.sleep(2)  # THIS BLOCKS! No await, no yield
    print("Done")

async def other_work():
    print("Other work starting")
    await asyncio.sleep(1)
    print("Other work done")

async def main():
    await asyncio.gather(blocking(), other_work())

asyncio.run(main())
```

Output:
```
Starting blocking operation
Done
Other work starting
Other work done
```

The "other work" cannot start until "blocking" finishes, even though we used `gather`. The `time.sleep()` blocks the entire event loop.

Common blocking operations:
- `time.sleep()` instead of `await asyncio.sleep()`
- Synchronous HTTP libraries (`requests` instead of `httpx.AsyncClient`)
- Synchronous database drivers
- CPU-intensive computation
- Synchronous file I/O

### How to Detect Blocking

Use asyncio's debug mode:

```python
import asyncio

async def main():
    # Enable debug mode
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    
    time.sleep(0.2)  # This will log a warning

asyncio.run(main())
```

Output includes: `Executing <Task pending ...> took 0.200 seconds`

### Fixing Blocking Code

**Solution 1: Use Async Libraries**

Replace sync libraries with async equivalents:

```python
# Instead of:
import requests
response = requests.get(url)

# Use:
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

Common async library alternatives:
- HTTP: `httpx`, `aiohttp` instead of `requests`
- Database: `asyncpg` instead of `psycopg2`, `aiomysql` instead of `mysql-connector`
- Redis: `aioredis` instead of `redis-py`
- Files: `aiofiles` instead of built-in `open()`

**Solution 2: run_in_executor for Sync Code**

When we must use sync code, run it in a thread pool:

```python
import asyncio
import time

def blocking_sync_code():
    time.sleep(2)
    return "result"

async def main():
    loop = asyncio.get_event_loop()
    
    # Run sync code in a thread pool
    result = await loop.run_in_executor(None, blocking_sync_code)
    print(result)

asyncio.run(main())
```

The sync code runs in a separate thread, so it does not block the event loop. Other async tasks can run while we wait.

```python
import asyncio
import requests  # Sync library

async def fetch_with_requests(url):
    """Use sync requests library without blocking event loop."""
    loop = asyncio.get_event_loop()
    
    # Run in thread pool
    response = await loop.run_in_executor(
        None,  # Use default executor
        requests.get,
        url
    )
    return response.json()

async def main():
    # These run concurrently despite using sync library
    results = await asyncio.gather(
        fetch_with_requests("https://api.example.com/1"),
        fetch_with_requests("https://api.example.com/2"),
        fetch_with_requests("https://api.example.com/3"),
    )
    print(results)
```

---

## 8. Practical Patterns

### Pattern 1: Concurrent API Calls with Rate Limiting

```python
import asyncio
import httpx

async def fetch_with_semaphore(client, url, semaphore):
    async with semaphore:  # Limit concurrent requests
        response = await client.get(url)
        return response.json()

async def fetch_all(urls, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_with_semaphore(client, url, semaphore)
            for url in urls
        ]
        results = await asyncio.gather(*tasks)
    
    return results

async def main():
    urls = [f"https://api.example.com/item/{i}" for i in range(100)]
    results = await fetch_all(urls, max_concurrent=10)
    print(f"Fetched {len(results)} items")

asyncio.run(main())
```

A semaphore limits how many coroutines can be in a section at once. This prevents overwhelming APIs with too many concurrent requests.

### Pattern 2: Retry with Exponential Backoff

```python
import asyncio
import httpx
import random

async def fetch_with_retry(client, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, httpx.RequestError) as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)

async def main():
    async with httpx.AsyncClient() as client:
        result = await fetch_with_retry(client, "https://api.example.com/data")
        print(result)

asyncio.run(main())
```

### Pattern 3: Timeout Wrapper

```python
import asyncio
from typing import TypeVar, Coroutine, Any

T = TypeVar('T')

async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: T = None
) -> T:
    """Run coroutine with timeout, return default on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        return default

async def main():
    async def slow_api_call():
        await asyncio.sleep(10)
        return "data"
    
    result = await with_timeout(slow_api_call(), timeout=2.0, default="timeout")
    print(result)  # "timeout"

asyncio.run(main())
```

### Pattern 4: Producer-Consumer with Queue

```python
import asyncio

async def producer(queue, n):
    """Produce items and put them in the queue."""
    for i in range(n):
        await asyncio.sleep(0.1)  # Simulate slow production
        await queue.put(f"item-{i}")
        print(f"Produced item-{i}")
    
    # Signal no more items
    await queue.put(None)

async def consumer(queue, name):
    """Consume items from the queue."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        
        await asyncio.sleep(0.2)  # Simulate slow processing
        print(f"{name} processed {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    
    # Start producer and multiple consumers
    await asyncio.gather(
        producer(queue, 10),
        consumer(queue, "Consumer-1"),
        consumer(queue, "Consumer-2"),
    )

asyncio.run(main())
```

### Pattern 5: Streaming Responses

For LLM applications, streaming is essential:

```python
import asyncio
import httpx

async def stream_llm_response(prompt):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            },
            headers={"Authorization": "Bearer $API_KEY"}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    # Process streaming chunk
                    chunk = line[6:]
                    if chunk != "[DONE]":
                        yield chunk

async def main():
    async for chunk in stream_llm_response("Tell me a joke"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Pattern 6: Running Async from Sync Code

Sometimes we need to call async code from a sync context:

```python
import asyncio

async def async_work():
    await asyncio.sleep(1)
    return "result"

def sync_function():
    # Option 1: Create new event loop (if none exists)
    result = asyncio.run(async_work())
    
    # Option 2: If loop exists but we're in sync code
    # loop = asyncio.get_event_loop()
    # result = loop.run_until_complete(async_work())
    
    return result
```

But be careful: `asyncio.run()` creates a new event loop each time. In frameworks like FastAPI that already have an event loop, use different patterns (see framework documentation).

---

## 9. Async Context Managers

Resources that need setup and cleanup work with `async with`:

```python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)  # Simulate async setup
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)  # Simulate async cleanup
        return False  # Don't suppress exceptions
    
    async def do_work(self):
        print("Working with resource")

async def main():
    async with AsyncResource() as resource:
        await resource.do_work()

asyncio.run(main())
```

Many async libraries provide async context managers:

```python
import httpx

async def main():
    async with httpx.AsyncClient() as client:  # Async context manager
        response = await client.get("https://example.com")
```

### Creating Async Context Managers with contextlib

```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def async_timer():
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.2f}s")

async def main():
    async with async_timer():
        await asyncio.sleep(1)
        # Prints: "Elapsed: 1.00s"

asyncio.run(main())
```

---

## 10. Async Iterators

Async iterators yield values one at a time, with async operations between:

```python
class AsyncCounter:
    def __init__(self, stop):
        self.stop = stop
        self.current = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current >= self.stop:
            raise StopAsyncIteration
        
        await asyncio.sleep(0.1)  # Async operation
        self.current += 1
        return self.current

async def main():
    async for num in AsyncCounter(5):
        print(num)

asyncio.run(main())
```

More commonly, we use async generators:

```python
async def async_range(start, stop):
    for i in range(start, stop):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_range(0, 5):
        print(num)
```

This is especially useful for streaming:

```python
async def stream_chunks(data, chunk_size):
    """Yield data in chunks with async processing between."""
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        await asyncio.sleep(0.01)  # Simulate processing
        yield chunk

async def main():
    data = list(range(100))
    async for chunk in stream_chunks(data, 10):
        print(f"Processing chunk: {chunk}")
```

---

## 11. Common Mistakes and Fixes

### Mistake 1: Forgetting to Await

```python
async def fetch_data():
    return "data"

async def main():
    result = fetch_data()  # Missing await!
    print(result)  # <coroutine object fetch_data at 0x...>
    
    # Python warns: "coroutine 'fetch_data' was never awaited"
```

Fix: Always await coroutines:
```python
result = await fetch_data()
```

### Mistake 2: Using Sync Libraries in Async Code

```python
import asyncio
import requests  # Sync library!

async def fetch(url):
    response = requests.get(url)  # Blocks the event loop!
    return response.json()
```

Fix: Use async libraries or run_in_executor:
```python
import httpx

async def fetch(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Mistake 3: Creating Tasks Without Awaiting

```python
async def background():
    await asyncio.sleep(1)
    print("Background done")

async def main():
    asyncio.create_task(background())  # Created but not awaited
    print("Main done")
    # Program might exit before background finishes!

asyncio.run(main())
```

Fix: Keep a reference and await before exiting:
```python
async def main():
    task = asyncio.create_task(background())
    print("Main done")
    await task  # Wait for it to finish
```

### Mistake 4: Sequential Awaits When Concurrent Is Possible

```python
async def main():
    # These run sequentially - slow!
    result1 = await fetch("url1")
    result2 = await fetch("url2")
    result3 = await fetch("url3")
```

Fix: Use gather for concurrent execution:
```python
async def main():
    result1, result2, result3 = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
```

### Mistake 5: Blocking in Async Functions

```python
import time

async def process():
    time.sleep(1)  # Blocks the event loop!
    return "done"
```

Fix: Use asyncio.sleep or run_in_executor:
```python
async def process():
    await asyncio.sleep(1)  # Non-blocking
    return "done"

# Or for sync code that must run:
async def process_sync_code():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_function)
    return result
```

---

## 12. When to Use Async vs Threading vs Multiprocessing

### Async Is Best For

- Many concurrent I/O operations (hundreds or thousands)
- Network-heavy applications (APIs, web scrapers, chat bots)
- When we need high concurrency with low overhead
- When the ecosystem provides async libraries

### Threading Is Better When

- We have mostly sync code and just want some concurrency
- We need to use sync libraries that cannot be easily replaced
- The number of concurrent operations is moderate (tens, not thousands)
- We want simpler code for simple use cases

### Multiprocessing Is For

- CPU-bound work (computation, not I/O)
- When we need true parallelism
- When we can divide work into independent chunks

### The Decision Process

```
What is the bottleneck?
├── Waiting for I/O (network, disk, etc.)
│   ├── Many concurrent operations (>100): async
│   ├── Fewer operations: threading or async
│   └── Must use sync libraries: threading with run_in_executor
└── CPU computation
    └── Use multiprocessing
```

For LLM applications, async is usually the right choice. We are making API calls, waiting for model responses, querying databases—all I/O operations where async excels.

---

## Summary

Async/await provides concurrency through cooperative multitasking in a single thread. The event loop is a scheduler that runs tasks until they await, then switches to other ready tasks.

Key concepts:
- `async def` creates a coroutine function
- `await` suspends the coroutine and yields to the event loop
- `asyncio.gather()` runs multiple coroutines concurrently
- `asyncio.create_task()` schedules a coroutine without blocking
- Never block the event loop with sync operations
- Use async libraries or `run_in_executor` for sync code

For LLM applications, async is ideal because:
- API calls are I/O-bound (waiting for responses)
- We often need to make many concurrent requests
- Streaming responses require non-blocking I/O
- Rate limiting is natural with semaphores

The event loop is not magic—it is just code that switches between tasks at await points. Understanding this model helps us write correct async code and debug issues when things go wrong.
