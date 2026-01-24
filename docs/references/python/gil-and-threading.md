# The GIL and Threading in Python

This document explains Python's Global Interpreter Lock and the threading module. We will understand what the GIL actually is, why it exists, when it matters, and when it does not. Then we will learn the threading module from the ground up, with practical patterns for building concurrent applications.

By the end of this guide, we will know when threading helps, when it hurts, and how to use it correctly for I/O-bound workloads like API calls, database queries, and file operations.

---

## 1. What Is the GIL, Really

The Global Interpreter Lock is the most misunderstood feature of Python. Before we can use threading effectively, we need to understand what the GIL actually is and what it is not.

### The Simple Definition

The GIL is a mutex—a mutual exclusion lock—that protects access to Python objects. At any given moment, only one thread can execute Python bytecode. If we have ten threads, only one of them is running Python code at a time. The others are waiting.

This sounds terrible. If only one thread runs at a time, what is the point of threading at all? This is the question everyone asks, and the answer is more nuanced than it first appears.

### Why Does the GIL Exist

Python's memory management relies on reference counting. Every Python object has a counter tracking how many references point to it. When we write `a = [1, 2, 3]`, the list object's reference count becomes 1. When we write `b = a`, the count becomes 2. When we delete `a`, the count drops to 1. When we delete `b`, the count drops to 0, and Python immediately frees the memory.

This reference counting happens constantly—every assignment, every function call, every attribute access modifies reference counts. And here is the problem: incrementing and decrementing a counter is not thread-safe. If two threads try to modify a reference count simultaneously, we get a race condition. The count becomes corrupted. Objects get freed while still in use, or never get freed at all.

The GIL solves this by ensuring only one thread touches Python objects at a time. It is a blunt solution, but it works. CPython, the standard Python implementation, chose simplicity and safety over maximum parallelism.

Other Python implementations make different choices. Jython (Python on the JVM) and IronPython (Python on .NET) do not have a GIL because they use garbage collectors instead of reference counting. PyPy has a GIL but is working on removing it. But if we are using standard Python—and most of us are—we have the GIL.

### When the GIL Releases

Here is the insight that makes threading useful despite the GIL: the GIL only protects Python bytecode execution. It releases during certain operations.

**I/O Operations**: When a thread makes a system call—reading a file, sending a network request, querying a database—it releases the GIL while waiting for the operating system to respond. During this time, other threads can run.

```python
import threading
import requests

def fetch(url):
    # GIL is released while waiting for network response
    response = requests.get(url)
    return response.status_code

# These can run concurrently because they spend most time waiting on I/O
threads = [threading.Thread(target=fetch, args=(url,)) for url in urls]
```

**C Extensions**: Libraries written in C can explicitly release the GIL while doing computation. NumPy releases the GIL during array operations. Pandas releases it during many data manipulations. When we call `numpy.dot(a, b)` on large arrays, the GIL is released during the actual matrix multiplication.

**Explicit Release**: Python's C API allows extension authors to release the GIL with `Py_BEGIN_ALLOW_THREADS` and reacquire it with `Py_END_ALLOW_THREADS`. This is how well-designed C extensions achieve parallelism.

### When the GIL Matters

The GIL only hurts when we have CPU-bound Python code running across multiple threads. Pure Python loops, mathematical calculations in Python, string processing—these hold the GIL continuously. If we try to parallelize them with threads, we get no speedup and sometimes even slowdowns due to context switching overhead.

```python
# This does NOT benefit from threading - CPU-bound Python code
def cpu_bound_work(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# Running this in 4 threads is slower than running it sequentially
# because threads contend for the GIL
```

### When the GIL Does Not Matter

For I/O-bound work, the GIL is largely irrelevant. If our threads spend most of their time waiting—waiting for HTTP responses, waiting for database queries, waiting for file reads—then the GIL releases during those waits, and other threads can run.

This is why threading is perfect for making concurrent API calls. Each thread spends 100 milliseconds waiting for a response and maybe 1 millisecond processing the result. During that 100 milliseconds of waiting, the GIL is released.

For LLM applications, this is exactly our situation. Calling the OpenAI API, fetching embeddings, querying vector databases—these are all I/O operations. Threading works well for them.

---

## 2. The Thread Switching Mechanism

Understanding how Python switches between threads helps predict behavior under load.

### The Check Interval

Python does not let a single thread hold the GIL forever. After executing a certain number of bytecode instructions, Python forces the thread to release the GIL and allows other threads to run.

In Python 3, this is controlled by `sys.getswitchinterval()`, which returns the interval in seconds (default is 0.005 seconds, or 5 milliseconds). Every 5 milliseconds, the running thread releases the GIL, and the operating system's thread scheduler decides which thread runs next.

```python
import sys

print(sys.getswitchinterval())  # 0.005

# We can change it, but rarely need to
sys.setswitchinterval(0.01)  # 10 milliseconds
```

### What Happens During a Switch

When the interval expires:

1. The running thread releases the GIL
2. The operating system picks which waiting thread to run next
3. That thread acquires the GIL
4. That thread executes until the next interval or until it voluntarily releases (for I/O)

This switching has overhead. If we have many threads fighting for the GIL, we spend time context switching instead of doing useful work. This is why threading is worse than useless for CPU-bound code—we add overhead without gaining parallelism.

### Observing GIL Contention

We can actually observe GIL contention. Here is an experiment:

```python
import threading
import time

def cpu_work():
    """Pure Python CPU work - holds GIL continuously."""
    total = 0
    for i in range(10_000_000):
        total += i
    return total

def measure(num_threads):
    threads = [threading.Thread(target=cpu_work) for _ in range(num_threads)]
    
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    
    return elapsed

# Running 1 thread
single = measure(1)
print(f"1 thread: {single:.2f}s")

# Running 4 threads (on a 4-core machine)
four = measure(4)
print(f"4 threads: {four:.2f}s")

# Expected output (approximate):
# 1 thread: 0.80s
# 4 threads: 3.20s  <- 4x SLOWER, not faster!
```

With CPU-bound work, four threads take four times as long as one thread. We are doing the same amount of work, but now with context switching overhead. This is the GIL in action.

Now compare with I/O-bound work:

```python
import threading
import time
import urllib.request

def io_work():
    """I/O-bound work - releases GIL while waiting."""
    urllib.request.urlopen("https://httpbin.org/delay/1")

def measure_io(num_threads):
    threads = [threading.Thread(target=io_work) for _ in range(num_threads)]
    
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    
    return elapsed

# Sequential: 4 requests take ~4 seconds
sequential = measure_io(1) * 4
print(f"Sequential: {sequential:.2f}s")

# Concurrent: 4 requests take ~1 second
concurrent = measure_io(4)
print(f"4 threads: {concurrent:.2f}s")  # ~1 second, not 4!
```

With I/O-bound work, four threads complete in roughly the same time as one thread making one request. The GIL is not the bottleneck—network latency is.

---

## 3. The Threading Module: Core Concepts

Now that we understand when threading helps, let us learn how to use it.

### Creating and Starting Threads

The basic pattern is straightforward:

```python
import threading

def worker(name):
    print(f"Worker {name} starting")
    # Do some work
    print(f"Worker {name} finished")

# Create a thread
t = threading.Thread(target=worker, args=("Alice",))

# Start it (this returns immediately)
t.start()

# Wait for it to complete
t.join()

print("Main thread continues after worker finishes")
```

When we call `t.start()`, Python creates a new operating system thread and begins executing our function in that thread. The main thread continues immediately—it does not wait.

When we call `t.join()`, the main thread blocks until the worker thread completes. This is how we synchronize—how we say "wait here until that thread is done."

### What Happens If We Forget to Join

If we do not call `join()`, the main thread continues without waiting. If the main thread finishes and exits, what happens to our worker threads?

By default, Python will wait for all non-daemon threads to complete before the process exits. So forgetting `join()` does not usually cause threads to be killed prematurely. But it does mean we might continue with code that depends on the thread's work before that work is done.

```python
results = []

def worker():
    # Simulate slow work
    time.sleep(1)
    results.append("done")

t = threading.Thread(target=worker)
t.start()
# Forgot to join!

print(results)  # Prints [] - the thread hasn't finished yet
```

Always join threads when we need their results or side effects.

### Daemon Threads

A daemon thread is a background thread that should not prevent the program from exiting. When only daemon threads remain, Python exits immediately without waiting for them.

```python
def background_work():
    while True:
        print("Background task running...")
        time.sleep(1)

# This thread runs in the background
t = threading.Thread(target=background_work, daemon=True)
t.start()

# Main thread does its work
time.sleep(3)
print("Main thread done")

# Process exits here - daemon thread is killed
```

Daemon threads are useful for background tasks that should not keep the program alive: logging, monitoring, periodic cleanup. But be careful—daemon threads can be killed at any moment, so they should not be doing work that requires cleanup.

### Getting Return Values from Threads

Threads cannot directly return values. The `target` function's return value is discarded. We need to use shared state or other mechanisms:

```python
import threading

# Approach 1: Store results in a shared list
results = []

def worker(n):
    result = n * n
    results.append(result)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(results)  # [0, 1, 4, 9, 16] - but order may vary!
```

There is a subtle bug here: appending to a list is thread-safe in CPython (due to the GIL), but the order of results is unpredictable. We will see better approaches with `Queue` and `ThreadPoolExecutor` later.

---

## 4. Race Conditions and Locks

When multiple threads access shared state, we risk race conditions—situations where the outcome depends on the unpredictable order of thread execution.

### A Classic Race Condition

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(100_000):
        counter += 1

threads = [threading.Thread(target=increment) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # Expected: 400,000. Actual: varies! Maybe 350,000
```

Wait, we said the GIL ensures only one thread runs at a time. How can we have a race condition?

The issue is that `counter += 1` is not a single operation. It expands to:

1. Read the current value of `counter`
2. Add 1 to it
3. Write the result back to `counter`

The GIL can release between any of these steps. Thread A reads `counter` as 100, then the GIL switches to Thread B, which reads `counter` as 100, increments to 101, writes 101. GIL switches back to Thread A, which adds 1 to its stale value of 100, writes 101. We lost an increment.

### Using Locks

A lock (mutex) ensures only one thread can access a critical section at a time:

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100_000):
        with lock:  # Only one thread can be here at a time
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # Always exactly 400,000
```

The `with lock:` syntax is equivalent to:

```python
lock.acquire()  # Wait until we get the lock
try:
    counter += 1
finally:
    lock.release()  # Always release, even if exception
```

### Lock vs RLock

A regular `Lock` can only be acquired once. If a thread that holds the lock tries to acquire it again, it deadlocks—it waits forever for itself to release the lock.

```python
lock = threading.Lock()

def outer():
    with lock:
        inner()  # Deadlock! We already hold the lock

def inner():
    with lock:  # Waits forever for outer() to release
        pass
```

An `RLock` (reentrant lock) can be acquired multiple times by the same thread:

```python
rlock = threading.RLock()

def outer():
    with rlock:
        inner()  # Works fine

def inner():
    with rlock:  # Same thread, can acquire again
        pass
```

When should we use `RLock`? When the same thread might need to enter a locked region from multiple call paths. This happens with recursive functions or when locked methods call other locked methods on the same object.

### Avoiding Deadlocks

A deadlock occurs when two threads each wait for a lock the other holds:

```python
lock_a = threading.Lock()
lock_b = threading.Lock()

def thread_1():
    with lock_a:
        time.sleep(0.1)  # Simulate work
        with lock_b:  # Waits for thread_2 to release lock_b
            pass

def thread_2():
    with lock_b:
        time.sleep(0.1)
        with lock_a:  # Waits for thread_1 to release lock_a
            pass

# Deadlock! Each thread holds one lock and waits for the other
```

The classic solution is lock ordering: always acquire locks in the same order across all threads:

```python
def thread_1():
    with lock_a:  # Always acquire A first
        with lock_b:
            pass

def thread_2():
    with lock_a:  # Same order: A first, then B
        with lock_b:
            pass
```

---

## 5. Thread Synchronization Primitives

Beyond locks, Python provides primitives for coordinating threads.

### Event: Signaling Between Threads

An `Event` is a simple flag that threads can wait on:

```python
import threading
import time

ready = threading.Event()

def worker():
    print("Worker waiting for signal...")
    ready.wait()  # Blocks until event is set
    print("Worker received signal, starting work")

def main():
    t = threading.Thread(target=worker)
    t.start()
    
    print("Main doing setup...")
    time.sleep(2)  # Simulate setup
    
    print("Main signaling worker")
    ready.set()  # Wake up all threads waiting on this event
    
    t.join()

main()
```

Events are useful for:
- Signaling that initialization is complete
- Coordinating startup across multiple workers
- Implementing simple start/stop patterns

### Condition: Wait for a Condition to Be True

A `Condition` combines a lock with the ability to wait for and signal conditions:

```python
import threading
import time

items = []
condition = threading.Condition()

def producer():
    for i in range(5):
        time.sleep(1)  # Simulate slow production
        with condition:
            items.append(i)
            print(f"Produced {i}")
            condition.notify()  # Wake up one waiting consumer

def consumer():
    while True:
        with condition:
            while not items:  # While no items available
                condition.wait()  # Release lock and wait for notify
            item = items.pop(0)
            print(f"Consumed {item}")
        
        if item == 4:  # Exit condition
            break

producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()
```

The key insight: `condition.wait()` releases the lock while waiting, then reacquires it when notified. This allows the producer to modify `items` while the consumer waits.

### Semaphore: Limiting Concurrent Access

A `Semaphore` allows a limited number of threads to access a resource:

```python
import threading
import time

# Allow at most 3 concurrent connections
connection_limit = threading.Semaphore(3)

def worker(id):
    print(f"Worker {id} waiting for connection")
    with connection_limit:
        print(f"Worker {id} got connection")
        time.sleep(2)  # Simulate using connection
        print(f"Worker {id} released connection")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

This is perfect for rate limiting—ensuring we do not overwhelm an API with too many concurrent requests.

---

## 6. Thread-Safe Data Structures: Queue

The `queue` module provides thread-safe data structures that handle all the locking internally.

### Basic Queue Usage

```python
import threading
import queue
import time

work_queue = queue.Queue()
results = []

def worker():
    while True:
        item = work_queue.get()  # Blocks until item available
        if item is None:  # Poison pill - signal to stop
            break
        
        # Process item
        result = item * 2
        results.append(result)
        
        work_queue.task_done()  # Signal that item is processed

# Start worker threads
workers = [threading.Thread(target=worker) for _ in range(4)]
for w in workers:
    w.start()

# Add work items
for i in range(20):
    work_queue.put(i)

# Wait for all items to be processed
work_queue.join()

# Stop workers
for _ in workers:
    work_queue.put(None)  # Send poison pills

for w in workers:
    w.join()

print(sorted(results))  # [0, 2, 4, 6, ..., 38]
```

Key methods:
- `put(item)`: Add item to queue (blocks if queue is full for bounded queues)
- `get()`: Remove and return item (blocks if queue is empty)
- `task_done()`: Signal that a retrieved item has been processed
- `join()`: Block until all items have been processed

### Queue Variants

```python
import queue

# FIFO queue (default)
q = queue.Queue()

# LIFO queue (stack)
q = queue.LifoQueue()

# Priority queue (lowest value first)
q = queue.PriorityQueue()
q.put((1, "low priority"))
q.put((0, "high priority"))
print(q.get())  # (0, "high priority")
```

---

## 7. ThreadPoolExecutor: The Modern Approach

For most use cases, `ThreadPoolExecutor` from `concurrent.futures` is cleaner than managing threads manually.

### Basic Usage

```python
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_data(url):
    time.sleep(1)  # Simulate network call
    return f"Data from {url}"

urls = ["url1", "url2", "url3", "url4", "url5"]

# Create a pool of 3 worker threads
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit all tasks
    futures = [executor.submit(fetch_data, url) for url in urls]
    
    # Get results as they complete
    for future in futures:
        result = future.result()  # Blocks until this future is done
        print(result)
```

The `with` statement ensures all threads are properly cleaned up when done.

### Using map() for Simpler Cases

When we want to apply a function to each item in a sequence:

```python
from concurrent.futures import ThreadPoolExecutor

def process(item):
    return item * 2

items = range(10)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process, items))

print(results)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

Results come back in the same order as inputs, even though processing may happen out of order.

### Handling Exceptions

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def risky_work(n):
    if n == 3:
        raise ValueError("Bad number!")
    return n * 2

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(risky_work, i): i for i in range(5)}
    
    for future in as_completed(futures):
        n = futures[future]
        try:
            result = future.result()
            print(f"Result for {n}: {result}")
        except Exception as e:
            print(f"Error for {n}: {e}")
```

`as_completed()` yields futures as they complete, regardless of submission order. This is useful when we want to process results as soon as they are ready.

### How Many Workers?

The `max_workers` parameter controls the pool size. For I/O-bound work:

```python
import os

# A common heuristic: 5x the number of CPU cores for I/O-bound work
workers = min(32, os.cpu_count() * 5)
```

More workers means more concurrent I/O operations, but also more memory usage and context switching. For API calls, we are often limited by rate limits rather than thread count.

For CPU-bound work, more threads does not help (because of the GIL). Use `ProcessPoolExecutor` instead.

---

## 8. Practical Patterns for LLM Applications

Let us apply threading to real scenarios.

### Pattern 1: Concurrent API Calls

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

def call_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

prompts = [
    "Explain Python's GIL in one sentence.",
    "What is a thread?",
    "Why use async/await?",
]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(call_llm, p): p for p in prompts}
    
    for future in as_completed(futures):
        prompt = futures[future]
        try:
            answer = future.result()
            print(f"Q: {prompt[:30]}...")
            print(f"A: {answer[:100]}...")
            print()
        except Exception as e:
            print(f"Error for '{prompt[:30]}...': {e}")
```

### Pattern 2: Rate-Limited Concurrent Requests

```python
from concurrent.futures import ThreadPoolExecutor
import threading
import time

class RateLimiter:
    def __init__(self, calls_per_second):
        self.interval = 1.0 / calls_per_second
        self.lock = threading.Lock()
        self.last_call = 0
    
    def wait(self):
        with self.lock:
            now = time.time()
            wait_time = self.last_call + self.interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call = time.time()

rate_limiter = RateLimiter(calls_per_second=5)  # Max 5 requests/second

def rate_limited_call(prompt):
    rate_limiter.wait()  # Wait for rate limit
    # Make API call
    return f"Response to: {prompt}"

prompts = [f"Question {i}" for i in range(20)]

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(rate_limited_call, prompts))
```

### Pattern 3: Producer-Consumer for Batch Processing

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

def process_documents(documents, num_workers=4):
    """Process documents with a producer-consumer pattern."""
    input_queue = queue.Queue()
    results = []
    results_lock = threading.Lock()
    
    def worker():
        while True:
            doc = input_queue.get()
            if doc is None:
                break
            
            # Process document (e.g., generate embedding)
            result = {"doc": doc, "embedding": [0.1, 0.2, 0.3]}
            
            with results_lock:
                results.append(result)
            
            input_queue.task_done()
    
    # Start workers
    workers = [threading.Thread(target=worker) for _ in range(num_workers)]
    for w in workers:
        w.start()
    
    # Add documents to queue
    for doc in documents:
        input_queue.put(doc)
    
    # Wait for processing to complete
    input_queue.join()
    
    # Stop workers
    for _ in workers:
        input_queue.put(None)
    for w in workers:
        w.join()
    
    return results

docs = [f"Document {i}" for i in range(100)]
results = process_documents(docs)
print(f"Processed {len(results)} documents")
```

### Pattern 4: Background Task Runner

```python
import threading
import queue
import time

class BackgroundTaskRunner:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
    
    def _worker(self):
        while True:
            task, args, kwargs = self.task_queue.get()
            try:
                task(*args, **kwargs)
            except Exception as e:
                print(f"Background task error: {e}")
            finally:
                self.task_queue.task_done()
    
    def submit(self, task, *args, **kwargs):
        self.task_queue.put((task, args, kwargs))
    
    def wait(self):
        self.task_queue.join()

# Usage
runner = BackgroundTaskRunner()

def log_to_database(message):
    time.sleep(0.1)  # Simulate DB write
    print(f"Logged: {message}")

# Fire and forget - returns immediately
runner.submit(log_to_database, "User logged in")
runner.submit(log_to_database, "API called")

# Optional: wait for all tasks to complete
runner.wait()
```

---

## 9. Debugging Threading Issues

Threading bugs are notoriously hard to reproduce and debug. Here are techniques that help.

### Logging Thread Activity

```python
import threading
import logging

# Configure logging to show thread names
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(threadName)s] %(message)s'
)

def worker():
    logging.debug("Starting work")
    # ... work ...
    logging.debug("Finished work")

t = threading.Thread(target=worker, name="Worker-1")
t.start()
```

### Naming Threads

```python
t = threading.Thread(target=worker, name="API-Fetcher-1")
# or
t.name = "API-Fetcher-1"
```

Named threads make logs and debugger output much more readable.

### Detecting Deadlocks

Python can dump thread stacks to help diagnose deadlocks:

```python
import faulthandler
import signal

# Enable thread dump on SIGUSR1 (Unix only)
faulthandler.register(signal.SIGUSR1)

# Now send `kill -USR1 <pid>` to dump all thread stacks
```

Or programmatically:

```python
import sys
import traceback

def dump_threads():
    for thread_id, frame in sys._current_frames().items():
        print(f"\nThread {thread_id}:")
        traceback.print_stack(frame)
```

---

## 10. When to Use Threading vs Other Approaches

### Threading Is Good For

- I/O-bound work: API calls, database queries, file operations
- Concurrent network requests (up to hundreds of concurrent connections)
- Background tasks that do not need true parallelism
- Existing sync code that we want to run concurrently

### Threading Is Not Good For

- CPU-bound work in Python (use `multiprocessing` instead)
- Very high concurrency (thousands of connections) — use `asyncio`
- When we need true parallelism for Python code (use `multiprocessing`)

### The Decision Tree

```
Is the work I/O-bound or CPU-bound?
├── I/O-bound: How many concurrent operations?
│   ├── < 100: Threading is fine
│   └── > 100: Consider asyncio
└── CPU-bound: What kind?
    ├── Pure Python: Use multiprocessing
    └── NumPy/Pandas/C extension: Threading works (GIL is released)
```

For LLM applications, we are almost always I/O-bound. Threading or asyncio both work well. Threading is simpler if we are already using sync libraries like `requests`. Asyncio is more efficient for very high concurrency.

---

## Summary

The GIL is not a bug—it is a design choice that simplifies Python's memory management at the cost of CPU parallelism. For I/O-bound work, which dominates LLM applications, the GIL is largely irrelevant because it releases during I/O waits.

Threading in Python:
- Use `ThreadPoolExecutor` for most cases—it handles thread lifecycle cleanly
- Use `Lock` to protect shared state from race conditions
- Use `Queue` for thread-safe data passing
- Name threads and use logging for debugging
- Do not use threading for CPU-bound Python code

The threading module is a tool. Like any tool, it works well when used for its intended purpose: running I/O-bound operations concurrently while we wait for external systems to respond.
