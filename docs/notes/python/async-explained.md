# Understanding Python Async/Await

The `async`/`await` syntax lets you write concurrent code that looks synchronous.

## The Core Insight

**Async doesn't make code faster** — it makes **waiting** more efficient.

When your program waits for I/O (network requests, file reads, database queries), the CPU sits idle. Async lets other tasks run during that waiting time.

## The Problem Async Solves

**Synchronous code (blocking):**

```python
import time

def fetch_data(url):
    time.sleep(2)  # Simulating network delay
    return f"Data from {url}"

# This takes 6 seconds total
result1 = fetch_data("api.com/1")  # Wait 2s
result2 = fetch_data("api.com/2")  # Wait 2s
result3 = fetch_data("api.com/3")  # Wait 2s
```

**Async code (concurrent):**

```python
import asyncio

async def fetch_data(url):
    await asyncio.sleep(2)  # Simulating network delay
    return f"Data from {url}"

# This takes 2 seconds total (all run concurrently)
async def main():
    results = await asyncio.gather(
        fetch_data("api.com/1"),
        fetch_data("api.com/2"),
        fetch_data("api.com/3")
    )
    
asyncio.run(main())
```

## Key Concepts

### `async def` — Defines a Coroutine

```python
async def my_function():
    return "Hello"
```

This creates a **coroutine** — a function that can pause and resume.

### `await` — Pauses Execution

```python
result = await some_async_function()
```

This says: "Pause here until `some_async_function()` completes. While waiting, let other tasks run."

### `asyncio.run()` — Runs the Event Loop

```python
asyncio.run(main())
```

This starts the async runtime. You need this to run async code.

## When to Use Async

### ✅ Good Use Cases

**Making multiple API calls:**
```python
async with aiohttp.ClientSession() as session:
    tasks = [fetch(session, url) for url in urls]
    results = await asyncio.gather(*tasks)
```

**Database queries** (with async drivers like asyncpg):
```python
async with pool.acquire() as conn:
    result = await conn.fetch("SELECT * FROM users")
```

**WebSocket connections:**
```python
async with websockets.connect(uri) as websocket:
    await websocket.send("Hello")
    response = await websocket.recv()
```

### ❌ Bad Use Cases

**CPU-bound tasks** (use `multiprocessing` instead):
```python
# Don't do this with async
def calculate_fibonacci(n):
    # Heavy computation, no I/O
    pass
```

**Simple scripts with one I/O operation:**
```python
# Async adds complexity for no benefit
response = requests.get(url)  # Just use requests
```

**Libraries without async support:**
```python
# Can't await this, it'll block the event loop
result = some_sync_library.call()  # ❌
```

## Common Mistakes

### 1. Blocking the Event Loop

```python
async def bad():
    time.sleep(5)  # ❌ Blocks everything!
    
async def good():
    await asyncio.sleep(5)  # ✅ Lets other tasks run
```

### 2. Forgetting `await`

```python
async def wrong():
    result = fetch_data()  # ❌ Returns a coroutine object
    
async def correct():
    result = await fetch_data()  # ✅ Gets the actual result
```

### 3. Mixing Sync and Async Code

```python
# ❌ Can't call async function from sync code directly
def sync_function():
    result = await async_function()  # SyntaxError!
    
# ✅ Need to use asyncio.run()
def sync_function():
    result = asyncio.run(async_function())
```

## Real Example: Fetching Multiple URLs

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
urls = [
    "https://api.example.com/users/1",
    "https://api.example.com/users/2",
    "https://api.example.com/users/3"
]

results = asyncio.run(fetch_all(urls))
```

## What I Misunderstood

**"Async makes Python multi-threaded"** — No. Async is single-threaded concurrency. Only one task executes at a time, but tasks can yield control while waiting.

**"I should make everything async"** — No. Async adds complexity. Only use it when you have significant I/O wait times.

**"Async is always faster"** — No. For CPU-bound tasks, async doesn't help. Use `multiprocessing` for true parallelism.

## Related Notes

- [Python Retry Decorator](../../snippets/python-retry-decorator.md) — Works with async too
- [Understanding the GIL](../python-gil.md) — Why async doesn't bypass the GIL
- [Async Database Queries](../databases/async-queries.md)

## Resources

- [Real Python: Async IO in Python](https://realpython.com/async-io-python/)
- [PEP 492: Coroutines with async/await](https://www.python.org/dev/peps/pep-0492/)
- [aiohttp Documentation](https://docs.aiohttp.org/)

---

*Last updated: January 20, 2026*