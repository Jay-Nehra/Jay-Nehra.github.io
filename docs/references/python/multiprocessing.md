# Multiprocessing in Python

This document explains Python's multiprocessing module—how to achieve true parallelism by running code in separate processes. We will understand when multiprocessing is the right choice, how it differs from threading, and how to use it effectively for CPU-bound workloads.

By the end of this guide, we will know how to parallelize CPU-intensive work, share data between processes safely, and avoid the common pitfalls that trip up newcomers.

---

## 1. Why Multiprocessing Exists

We established in the threading guide that the GIL prevents true parallelism for CPU-bound Python code. If we have a computation that takes 10 seconds and we want to run four of them, threading gives us 40 seconds (plus overhead), not 10 seconds.

Multiprocessing solves this by running code in entirely separate processes. Each process has its own Python interpreter, its own memory space, and its own GIL. Four processes can truly run in parallel on four CPU cores.

### Process vs Thread: The Fundamental Difference

A thread shares memory with its parent process. All threads in a process see the same global variables, the same objects, the same everything. This makes communication easy but creates race conditions.

A process has completely separate memory. A child process gets a copy of the parent's memory at the moment of creation, but after that, they are independent. Changes in one process are invisible to the other. This eliminates race conditions but makes communication harder.

```python
import threading
import multiprocessing

shared_value = 0

def increment_thread():
    global shared_value
    shared_value += 1

def increment_process():
    global shared_value
    shared_value += 1

# Threading: shared memory
t = threading.Thread(target=increment_thread)
t.start()
t.join()
print(f"After thread: {shared_value}")  # 1

# Multiprocessing: separate memory
p = multiprocessing.Process(target=increment_process)
p.start()
p.join()
print(f"After process: {shared_value}")  # Still 1! Process had its own copy
```

This is the fundamental insight: processes do not share memory by default. If we want to share data, we must do so explicitly.

### When to Use Multiprocessing

Multiprocessing is the right choice when:

1. **CPU-bound Python code**: Pure Python loops, mathematical calculations, data transformations that do not use C extensions
2. **Work that can be parallelized**: Independent tasks that do not need to share state
3. **We have multiple CPU cores**: Multiprocessing cannot speed up single-core machines

Multiprocessing is not ideal when:

1. **I/O-bound work**: Threading or asyncio is more efficient (less overhead)
2. **Shared state is essential**: Inter-process communication has significant overhead
3. **Startup time matters**: Creating processes is slower than creating threads
4. **Memory is limited**: Each process duplicates memory

### The Overhead Question

Creating a process is expensive. The operating system must:
- Copy the parent's memory space (or set up copy-on-write)
- Create new file descriptor tables
- Set up new process scheduling structures

This takes milliseconds, not microseconds. For very short tasks, the overhead dominates. Multiprocessing shines when tasks are long enough that parallelism outweighs startup cost.

---

## 2. Basic Process Creation

The `multiprocessing` module mirrors the `threading` module's API, making it familiar.

### Creating and Starting Processes

```python
import multiprocessing
import os

def worker(name):
    pid = os.getpid()
    print(f"Worker {name} running in process {pid}")

if __name__ == "__main__":
    print(f"Main process: {os.getpid()}")
    
    p = multiprocessing.Process(target=worker, args=("Alice",))
    p.start()  # Starts the process
    p.join()   # Waits for it to finish
    
    print("Main process continues")
```

Output:
```
Main process: 12345
Worker Alice running in process 12346
Main process continues
```

Notice the different process IDs. The worker runs in a completely separate process.

### The `if __name__ == "__main__"` Requirement

On Windows and macOS (with "spawn" start method), Python creates new processes by importing the main module fresh. Without the `if __name__ == "__main__"` guard, the child process would try to create more child processes during import, leading to infinite recursion.

```python
import multiprocessing

def worker():
    print("Working")

# This creates processes at import time - BAD!
# p = multiprocessing.Process(target=worker)
# p.start()

# This only creates processes when run as main script - GOOD!
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
```

This guard is mandatory for multiprocessing on Windows. On Linux (with "fork"), it often works without it, but using the guard is a good habit for cross-platform code.

### Start Methods

Python supports different ways of creating child processes:

```python
import multiprocessing

# See current start method
print(multiprocessing.get_start_method())  # 'fork', 'spawn', or 'forkserver'

# Set start method (must be done before creating any processes)
multiprocessing.set_start_method('spawn')
```

**fork** (Linux default): Child process is a copy of the parent at the moment of forking. Fast but can cause issues with threads and file handles.

**spawn** (Windows/macOS default): Child process starts fresh and imports the main module. Slower but safer.

**forkserver**: A compromise—a server process is forked once, and new processes are forked from that server.

For most applications, the default is fine. But if we are mixing threading with multiprocessing, "spawn" is safer.

---

## 3. Getting Results from Processes

Unlike threads, processes cannot share return values through shared memory. We need explicit mechanisms.

### Using Queue for Results

```python
import multiprocessing

def compute_square(n, result_queue):
    result = n * n
    result_queue.put((n, result))

if __name__ == "__main__":
    result_queue = multiprocessing.Queue()
    
    processes = []
    for i in range(5):
        p = multiprocessing.Process(
            target=compute_square, 
            args=(i, result_queue)
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    # Collect results
    results = {}
    while not result_queue.empty():
        n, result = result_queue.get()
        results[n] = result
    
    print(results)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

The `Queue` is a thread-and-process-safe data structure. Data put into it by child processes can be retrieved by the parent.

### Using Pipe for Two-Way Communication

For communication between exactly two processes, `Pipe` is more efficient than `Queue`:

```python
import multiprocessing

def worker(conn):
    # Receive data from parent
    data = conn.recv()
    
    # Process it
    result = data.upper()
    
    # Send result back
    conn.send(result)
    conn.close()

if __name__ == "__main__":
    parent_conn, child_conn = multiprocessing.Pipe()
    
    p = multiprocessing.Process(target=worker, args=(child_conn,))
    p.start()
    
    parent_conn.send("hello")
    print(parent_conn.recv())  # "HELLO"
    
    p.join()
```

A pipe has two ends. Data sent on one end appears on the other.

---

## 4. Pool: The Simple Way to Parallelize

For most use cases, `Pool` is the easiest way to parallelize work. It manages a pool of worker processes and distributes tasks to them.

### Basic Pool Usage

```python
import multiprocessing
import time

def slow_square(n):
    time.sleep(1)  # Simulate slow computation
    return n * n

if __name__ == "__main__":
    # Create a pool of 4 workers
    with multiprocessing.Pool(processes=4) as pool:
        # Map function over inputs
        numbers = [1, 2, 3, 4, 5, 6, 7, 8]
        results = pool.map(slow_square, numbers)
    
    print(results)  # [1, 4, 9, 16, 25, 36, 49, 64]
```

With 4 workers and 8 one-second tasks, this takes about 2 seconds instead of 8.

### Pool Methods

**map(func, iterable)**: Like built-in `map()`, but parallel. Blocks until all results are ready. Returns results in input order.

```python
results = pool.map(func, [1, 2, 3, 4])
# Returns [func(1), func(2), func(3), func(4)]
```

**imap(func, iterable)**: Returns an iterator. Results come as they complete (but still in order). Memory-efficient for large iterables.

```python
for result in pool.imap(func, huge_list):
    process(result)  # Process results one at a time
```

**imap_unordered(func, iterable)**: Like `imap`, but results may come in any order. Fastest option when order does not matter.

```python
for result in pool.imap_unordered(func, huge_list):
    process(result)  # Get results as soon as they're ready
```

**apply_async(func, args)**: Submit a single task asynchronously. Returns an `AsyncResult` object.

```python
async_result = pool.apply_async(func, (arg1, arg2))
# Do other work...
result = async_result.get()  # Block until result is ready
```

**starmap(func, iterable)**: Like `map`, but unpacks arguments. Useful when function takes multiple arguments.

```python
def add(a, b):
    return a + b

results = pool.starmap(add, [(1, 2), (3, 4), (5, 6)])
# Returns [3, 7, 11]
```

### How Many Processes?

```python
import multiprocessing
import os

# Default: number of CPUs
default_workers = multiprocessing.cpu_count()
print(f"CPU count: {default_workers}")

# For CPU-bound work: use CPU count
with multiprocessing.Pool() as pool:  # Uses cpu_count() by default
    pass

# For mixed workloads: experiment to find optimal
with multiprocessing.Pool(processes=os.cpu_count() * 2) as pool:
    pass
```

For pure CPU-bound work, using more processes than CPU cores gives no benefit and adds overhead. For mixed I/O and CPU work, more processes can help hide I/O latency.

### Handling Exceptions

```python
import multiprocessing

def risky_work(n):
    if n == 3:
        raise ValueError("I don't like 3!")
    return n * 2

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        try:
            results = pool.map(risky_work, range(5))
        except ValueError as e:
            print(f"Error: {e}")  # Catches exception from worker
```

With `map`, an exception in any worker stops everything and re-raises in the parent. With `imap`, we can handle exceptions per-item:

```python
with multiprocessing.Pool(4) as pool:
    results = pool.imap(risky_work, range(5))
    for i in range(5):
        try:
            print(results.next())
        except ValueError as e:
            print(f"Item failed: {e}")
```

---

## 5. ProcessPoolExecutor: The Modern Interface

`concurrent.futures.ProcessPoolExecutor` provides a more modern interface that mirrors `ThreadPoolExecutor`:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def slow_work(n):
    time.sleep(1)
    return n * n

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit individual tasks
        futures = [executor.submit(slow_work, i) for i in range(8)]
        
        # Get results as they complete
        for future in as_completed(futures):
            result = future.result()
            print(f"Got result: {result}")
```

The advantage of `ProcessPoolExecutor` is API consistency with `ThreadPoolExecutor`. We can switch between them by changing one import:

```python
# For I/O-bound work
from concurrent.futures import ThreadPoolExecutor as Executor

# For CPU-bound work
from concurrent.futures import ProcessPoolExecutor as Executor

with Executor(max_workers=4) as executor:
    results = list(executor.map(func, items))
```

---

## 6. Sharing State Between Processes

Sometimes we need processes to share data. Multiprocessing provides several mechanisms, each with tradeoffs.

### Value and Array: Shared Memory

For simple values and arrays, we can use shared memory:

```python
import multiprocessing

def increment(shared_counter, lock):
    for _ in range(10000):
        with lock:
            shared_counter.value += 1

if __name__ == "__main__":
    # 'i' = signed integer
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    
    processes = [
        multiprocessing.Process(target=increment, args=(counter, lock))
        for _ in range(4)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(counter.value)  # 40000
```

Type codes for Value:
- `'i'`: signed int
- `'d'`: double float
- `'c'`: char

For arrays:

```python
import multiprocessing

def worker(shared_array, index):
    shared_array[index] = index * index

if __name__ == "__main__":
    # Array of 5 signed integers
    arr = multiprocessing.Array('i', 5)
    
    processes = [
        multiprocessing.Process(target=worker, args=(arr, i))
        for i in range(5)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(list(arr))  # [0, 1, 4, 9, 16]
```

### Manager: Shared Python Objects

For more complex data structures, use a `Manager`:

```python
import multiprocessing

def worker(shared_dict, shared_list, key):
    shared_dict[key] = key * 2
    shared_list.append(key)

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_list = manager.list()
    
    processes = [
        multiprocessing.Process(target=worker, args=(shared_dict, shared_list, i))
        for i in range(5)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(dict(shared_dict))  # {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
    print(list(shared_list))  # [0, 1, 2, 3, 4] (order may vary)
```

Manager objects are synchronized—they handle locking internally. But they are slower than Value/Array because they use a separate server process for coordination.

### When to Share vs When to Pass

Sharing state adds complexity and overhead. Often it is better to:

1. Pass input data to workers
2. Let workers return results
3. Combine results in the parent

```python
# Instead of shared state
def process_chunk(chunk):
    """Process independently, return result."""
    return sum(x * x for x in chunk)

if __name__ == "__main__":
    data = list(range(1_000_000))
    chunk_size = len(data) // 4
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    with multiprocessing.Pool(4) as pool:
        partial_sums = pool.map(process_chunk, chunks)
    
    total = sum(partial_sums)
    print(total)
```

This "map-reduce" pattern avoids shared state entirely.

---

## 7. Serialization and Pickling

When we pass data to a child process, Python serializes it using `pickle`. The child deserializes it. This has important implications.

### What Can Be Pickled

Most Python objects can be pickled:
- Basic types: int, float, str, bytes, None, bool
- Collections: list, tuple, dict, set
- Classes and instances (with caveats)
- Functions defined at module level

### What Cannot Be Pickled

- Lambda functions
- Nested functions (functions defined inside other functions)
- Open file handles, sockets, database connections
- Thread locks, semaphores
- Some C extension objects

```python
import multiprocessing

# This fails - lambda cannot be pickled
# pool.map(lambda x: x * 2, [1, 2, 3])

# This works - module-level function
def double(x):
    return x * 2

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        results = pool.map(double, [1, 2, 3])
```

### Pickling Overhead

Large objects take time to pickle and unpickle. If we are passing megabytes of data to workers, serialization can dominate runtime.

```python
import multiprocessing
import numpy as np
import time

def process_array(arr):
    return arr.sum()

if __name__ == "__main__":
    # Large array - expensive to pickle
    large_array = np.random.rand(10_000_000)
    
    start = time.perf_counter()
    with multiprocessing.Pool(4) as pool:
        # This pickles large_array 4 times (once per worker)
        results = pool.map(process_array, [large_array] * 4)
    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.2f}s")  # Much slower than expected
```

For large data, consider:
- Using shared memory (numpy arrays with `multiprocessing.shared_memory`)
- Passing file paths instead of data
- Chunking data so each worker loads its own portion

---

## 8. Practical Patterns

### Pattern 1: Parallel Data Processing

```python
import multiprocessing
from pathlib import Path
import json

def process_file(filepath):
    """Process a single file and return results."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Process the data
    result = {
        "file": filepath.name,
        "record_count": len(data),
        "processed": True
    }
    return result

if __name__ == "__main__":
    files = list(Path("data/").glob("*.json"))
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, files)
    
    print(f"Processed {len(results)} files")
```

### Pattern 2: Batch Embedding Generation

```python
import multiprocessing
from sentence_transformers import SentenceTransformer

def generate_embeddings(texts):
    """Generate embeddings for a batch of texts."""
    # Load model in each worker (loaded once per process)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def chunk_list(lst, n):
    """Split list into n roughly equal chunks."""
    chunk_size = len(lst) // n + 1
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

if __name__ == "__main__":
    documents = [f"Document {i}" for i in range(10000)]
    num_workers = 4
    chunks = chunk_list(documents, num_workers)
    
    with multiprocessing.Pool(num_workers) as pool:
        embedding_batches = pool.map(generate_embeddings, chunks)
    
    # Combine results
    import numpy as np
    all_embeddings = np.vstack(embedding_batches)
    print(f"Generated {len(all_embeddings)} embeddings")
```

Note: The model is loaded once per worker process. This is intentional—we pay the loading cost once per process, not once per document.

### Pattern 3: CPU-Intensive Computation

```python
import multiprocessing
import math

def compute_primes(start, end):
    """Find primes in range [start, end)."""
    primes = []
    for n in range(max(2, start), end):
        if all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1)):
            primes.append(n)
    return primes

if __name__ == "__main__":
    ranges = [(0, 25000), (25000, 50000), (50000, 75000), (75000, 100000)]
    
    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(compute_primes, ranges)
    
    all_primes = [p for primes in results for p in primes]
    print(f"Found {len(all_primes)} primes")
```

### Pattern 4: Initializing Workers Once

Sometimes workers need expensive initialization (loading models, opening connections). Use `initializer`:

```python
import multiprocessing

# Global variable in each worker
model = None

def init_worker():
    """Called once per worker process."""
    global model
    print(f"Initializing worker {multiprocessing.current_process().name}")
    # Expensive initialization
    model = load_heavy_model()

def process(item):
    """Uses the globally initialized model."""
    global model
    return model.predict(item)

if __name__ == "__main__":
    with multiprocessing.Pool(
        processes=4,
        initializer=init_worker
    ) as pool:
        results = pool.map(process, items)
```

The `initializer` function runs once when each worker process starts, not for each task.

---

## 9. Common Pitfalls and How to Avoid Them

### Pitfall 1: Forgetting `if __name__ == "__main__"`

```python
import multiprocessing

def worker():
    pass

# WRONG - causes infinite process creation on Windows/spawn
p = multiprocessing.Process(target=worker)
p.start()

# RIGHT
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
```

### Pitfall 2: Expecting Shared Memory

```python
import multiprocessing

data = []  # This is NOT shared!

def worker(x):
    data.append(x)  # Appends to worker's copy, not parent's

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        pool.map(worker, range(10))
    
    print(data)  # [] - empty! Workers had their own copies
```

Fix: Return results instead of modifying shared state:

```python
def worker(x):
    return x * 2

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        results = pool.map(worker, range(10))
    print(results)  # [0, 2, 4, ..., 18]
```

### Pitfall 3: Lambda Functions

```python
import multiprocessing

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        # WRONG - lambda cannot be pickled
        # results = pool.map(lambda x: x * 2, range(10))
        pass
```

Fix: Use module-level functions:

```python
def double(x):
    return x * 2

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        results = pool.map(double, range(10))
```

### Pitfall 4: Not Joining Processes

```python
import multiprocessing

def worker():
    # Long-running work
    pass

if __name__ == "__main__":
    processes = [multiprocessing.Process(target=worker) for _ in range(4)]
    for p in processes:
        p.start()
    
    # WRONG - main process might exit before workers finish
    print("Done")
    
    # RIGHT - wait for all workers
    for p in processes:
        p.join()
    print("Done")
```

### Pitfall 5: Process Cleanup Failures

```python
import multiprocessing

if __name__ == "__main__":
    pool = multiprocessing.Pool(4)
    try:
        results = pool.map(some_function, data)
    finally:
        pool.close()  # Stop accepting new tasks
        pool.join()   # Wait for workers to finish
```

Better: Use context manager for automatic cleanup:

```python
if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        results = pool.map(some_function, data)
    # Pool is automatically closed and joined
```

---

## 10. Debugging Multiprocessing Code

Debugging multiprocessing is harder than single-threaded code because:
- Print statements may be interleaved
- Debuggers cannot easily attach to child processes
- Exceptions in workers may be swallowed

### Logging

Use multiprocessing-aware logging:

```python
import multiprocessing
import logging

def worker_init():
    """Configure logging for worker processes."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(processName)s: %(message)s'
    )

def worker(item):
    logging.info(f"Processing {item}")
    return item * 2

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(processName)s: %(message)s'
    )
    
    with multiprocessing.Pool(4, initializer=worker_init) as pool:
        results = pool.map(worker, range(10))
```

### Debugging Strategy

1. **Test with 1 process first**: Errors are easier to see without parallelism
2. **Add extensive logging**: Log inputs, outputs, and errors
3. **Catch and log exceptions in workers**:

```python
def safe_worker(item):
    try:
        return actual_work(item)
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}
```

4. **Use `maxtasksperchild` to limit worker lifetime**:

```python
# Each worker handles at most 10 tasks before being replaced
with multiprocessing.Pool(4, maxtasksperchild=10) as pool:
    results = pool.map(worker, items)
```

This helps catch memory leaks and resource cleanup issues.

---

## 11. Threading vs Multiprocessing: The Decision

| Aspect | Threading | Multiprocessing |
|--------|-----------|-----------------|
| Best for | I/O-bound work | CPU-bound work |
| Memory | Shared | Separate |
| Startup cost | Low | High |
| Communication | Easy (shared memory) | Requires IPC |
| GIL impact | Limited by GIL | Bypasses GIL |
| Debugging | Easier | Harder |

### Decision Tree

```
What kind of work?
├── I/O-bound (API calls, file I/O, DB queries)
│   └── Use threading or asyncio
├── CPU-bound
│   ├── Pure Python computation
│   │   └── Use multiprocessing
│   └── NumPy/pandas/C extensions
│       └── Threading works (GIL released during C code)
└── Mixed
    └── Consider both or hybrid approaches
```

### Hybrid Approach

Sometimes we need both:

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def io_work(item):
    # I/O-bound work in threads
    pass

def cpu_work(item):
    # CPU-bound work in processes
    pass

if __name__ == "__main__":
    # Stage 1: Parallel I/O
    with ThreadPoolExecutor(max_workers=10) as executor:
        io_results = list(executor.map(io_work, items))
    
    # Stage 2: Parallel CPU processing
    with ProcessPoolExecutor(max_workers=4) as executor:
        cpu_results = list(executor.map(cpu_work, io_results))
```

---

## Summary

Multiprocessing gives us true parallelism by running code in separate processes, bypassing the GIL. It is the right choice for CPU-bound Python code that can be split into independent tasks.

Key points:
- Each process has its own memory—sharing requires explicit mechanisms
- Use `Pool` or `ProcessPoolExecutor` for most use cases
- `if __name__ == "__main__"` is mandatory on Windows
- Data must be picklable to pass between processes
- Return results instead of modifying shared state when possible
- The overhead of process creation and IPC means multiprocessing is best for substantial tasks

For LLM applications, multiprocessing is useful for:
- Batch embedding generation
- Parallel data preprocessing
- CPU-intensive text processing
- Any computation that takes seconds per item

For network I/O (API calls, database queries), stick with threading or asyncio—they have less overhead and the GIL is not a bottleneck.
