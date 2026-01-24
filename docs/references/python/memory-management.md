# Memory Management in Python

This document explains how Python manages memory—the mechanisms that allocate, track, and free objects. We will understand reference counting, garbage collection, and why memory leaks happen even in a garbage-collected language.

By the end of this guide, we will know how to debug memory issues, profile memory usage, and write code that does not leak in long-running applications.

---

## 1. Why Memory Management Matters

Python handles memory automatically. We create objects, use them, and eventually they disappear. This sounds like magic, but understanding the mechanism matters for several reasons.

**Long-running servers**: A web server that handles millions of requests cannot afford memory leaks. A slow leak of 1KB per request becomes 1GB after a million requests.

**Large data processing**: When we load a 10GB dataset, we need to understand when and how that memory gets freed.

**Performance tuning**: Knowing why objects persist helps us optimize memory-intensive applications.

**Debugging**: When memory grows unexpectedly, we need to know where to look.

### The Basic Contract

When we create an object in Python, memory is allocated for it. When the object is no longer needed, memory is freed. The key question is: how does Python know when an object is "no longer needed"?

Python uses two complementary mechanisms:
1. **Reference counting**: The primary mechanism, handles most cases immediately
2. **Garbage collection**: A backup for cases reference counting cannot handle

---

## 2. Reference Counting

Every Python object has a reference count—an integer tracking how many references point to it. When the count drops to zero, the object is immediately deallocated.

### How References Work

```python
a = [1, 2, 3]  # Create list, refcount = 1
b = a          # b points to same list, refcount = 2
c = a          # c also points to it, refcount = 3

del b          # Remove one reference, refcount = 2
c = None       # c no longer points to list, refcount = 1
# a still points to the list, so it stays alive

del a          # refcount = 0, list is freed immediately
```

### Creating References

Many operations create references without us realizing:

```python
my_list = [1, 2, 3]           # 1 reference (the variable)
another = my_list             # 2 references
stored = {"data": my_list}    # 3 references (dict value)
passed_to_function(my_list)   # 4 references inside function

# When function returns, its reference is released
# When we reassign 'another', that reference is released
# And so on...
```

Containers (lists, dicts, sets) hold references to their contents:

```python
container = []
obj = {"data": 123}
container.append(obj)  # container now holds a reference to obj

del obj  # obj variable gone, but object lives in container
print(container[0])  # Still accessible!
```

### Observing Reference Counts

```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (a + temporary reference from getrefcount)

b = a
print(sys.getrefcount(a))  # 3

del b
print(sys.getrefcount(a))  # 2
```

The count is always 1 higher than expected because `getrefcount` itself creates a temporary reference.

### Immediate Deallocation

Reference counting's big advantage is immediacy. When the count hits zero, memory is freed right away:

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Allocated: {name}")
    
    def __del__(self):
        print(f"Freed: {self.name}")

def demo():
    r = Resource("test")
    print("Using resource")
    # r goes out of scope here

demo()
print("After function")
```

Output:
```
Allocated: test
Using resource
Freed: test
After function
```

The resource is freed immediately when the function returns, not at some later garbage collection cycle.

---

## 3. The Circular Reference Problem

Reference counting has a fatal flaw: it cannot handle circular references.

### What Are Circular References

```python
a = []
b = []
a.append(b)  # a references b
b.append(a)  # b references a

# Now:
# a: refcount = 2 (variable a + b's list)
# b: refcount = 2 (variable b + a's list)

del a  # a's refcount: 2 -> 1 (still > 0!)
del b  # b's refcount: 2 -> 1 (still > 0!)

# Both objects still have refcount 1
# But they're unreachable from our code
# Reference counting cannot free them
```

This is a memory leak. The objects are unreachable but not freed.

Circular references are common in real code:

```python
class Node:
    def __init__(self):
        self.parent = None
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # Creates a cycle!

root = Node()
child = Node()
root.add_child(child)

# root <-> child form a cycle
# When we delete root and child variables, the objects persist
```

### Why This Matters in Practice

```python
class Handler:
    def __init__(self):
        self.callback = None
    
    def set_callback(self, fn):
        self.callback = fn

def create_handler():
    handler = Handler()
    
    def callback():
        print(handler.data)  # Closure captures handler
    
    handler.set_callback(callback)  # Handler references callback
    handler.data = "test"
    return handler

# callback references handler, handler references callback
# This is a common pattern that creates cycles
```

---

## 4. Garbage Collection

Python's garbage collector exists to handle circular references. It periodically scans for groups of objects that reference each other but are unreachable from the program.

### How Garbage Collection Works

The GC uses a generational algorithm with three generations (0, 1, 2):

- **Generation 0**: New objects go here
- **Generation 1**: Objects that survived one GC cycle
- **Generation 2**: Long-lived objects that survived multiple cycles

Young objects are collected frequently (they tend to die young). Old objects are collected rarely (if they have lived this long, they will probably live longer).

The GC does not scan all objects every time. It focuses on young objects, occasionally scanning older generations.

### Viewing GC Statistics

```python
import gc

# Get counts of objects in each generation
print(gc.get_count())  # (700, 10, 2) - objects in gen 0, 1, 2

# Get thresholds that trigger collection
print(gc.get_threshold())  # (700, 10, 10)
# When gen 0 reaches 700 objects, collect gen 0
# When gen 0 has been collected 10 times, collect gen 1
# When gen 1 has been collected 10 times, collect gen 2
```

### Manual Garbage Collection

```python
import gc

# Force a full collection of all generations
gc.collect()

# Collect specific generations
gc.collect(0)  # Just generation 0
gc.collect(1)  # Generations 0 and 1
gc.collect(2)  # All generations

# Disable automatic GC (not usually recommended)
gc.disable()
gc.enable()
```

### Finding Circular References

```python
import gc

class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None

# Create a cycle
a = Node("a")
b = Node("b")
a.ref = b
b.ref = a

del a, b

# Before collection, enable debug to see what's found
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()

# gc.garbage contains objects that couldn't be freed
# (usually because they have __del__ methods)
print(gc.garbage)
```

---

## 5. The __del__ Method

The `__del__` method is called when an object is about to be deallocated. But it has quirks that cause problems.

### When __del__ Is Called

```python
class Resource:
    def __del__(self):
        print("Destructor called")

r = Resource()
del r  # Destructor called immediately (if refcount hits 0)
```

### The Problem with __del__ and Cycles

```python
import gc

class Problematic:
    def __init__(self, name):
        self.name = name
        self.ref = None
    
    def __del__(self):
        # If this accesses self.ref, and ref is being destroyed too,
        # we don't know which order they're destroyed in
        print(f"Destroying {self.name}")

a = Problematic("a")
b = Problematic("b")
a.ref = b
b.ref = a

del a, b
gc.collect()  # May not collect properly!
```

When objects with `__del__` form cycles, the GC cannot safely destroy them because it does not know which order to call the destructors. In older Python (pre-3.4), these ended up in `gc.garbage`. In Python 3.4+, the GC tries harder but issues can still occur.

### Best Practice: Use Context Managers Instead

```python
# Instead of relying on __del__
class Resource:
    def __del__(self):
        self.cleanup()  # Unreliable timing!
    
    def cleanup(self):
        print("Cleaning up")

# Use context managers
class Resource:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()  # Guaranteed to be called
    
    def cleanup(self):
        print("Cleaning up")

with Resource() as r:
    # Use resource
    pass
# cleanup() called here, guaranteed
```

---

## 6. Weak References

Weak references allow us to reference an object without increasing its reference count. The object can be garbage collected even while weak references to it exist.

### Basic Weak References

```python
import weakref

class Data:
    def __init__(self, value):
        self.value = value

obj = Data(42)
weak = weakref.ref(obj)

print(weak())  # Data object - dereference the weak ref
print(weak().value)  # 42

del obj  # Object can be freed, weak ref doesn't keep it alive
print(weak())  # None - object is gone
```

### WeakValueDictionary for Caches

```python
import weakref

class ExpensiveObject:
    def __init__(self, key):
        self.key = key
        print(f"Creating expensive object {key}")

# Normal dict would keep objects alive forever
# cache = {}

# WeakValueDictionary lets objects be collected when not used elsewhere
cache = weakref.WeakValueDictionary()

def get_expensive(key):
    if key in cache:
        print(f"Cache hit: {key}")
        return cache[key]
    
    obj = ExpensiveObject(key)
    cache[key] = obj
    return obj

# Use the cache
obj1 = get_expensive("a")  # Creates new object
obj2 = get_expensive("a")  # Cache hit

del obj1, obj2  # Objects can be collected now
# Even though they're in cache, it's a weak reference

import gc
gc.collect()

obj3 = get_expensive("a")  # Creates new object (cache was cleared)
```

### Callbacks on Object Death

```python
import weakref

def callback(weak_ref):
    print("Object was destroyed!")

obj = object()
weak = weakref.ref(obj, callback)

del obj  # Prints "Object was destroyed!"
```

This is useful for cleanup when we cannot use context managers.

---

## 7. Common Memory Leak Patterns

Even with garbage collection, Python programs can leak memory. Understanding common patterns helps avoid and debug leaks.

### Leak 1: Growing Collections

```python
# Memory grows forever
results = []

def process_request(data):
    result = expensive_computation(data)
    results.append(result)  # Never cleared!
    return result

# Fix: Use bounded collections or clear periodically
from collections import deque
results = deque(maxlen=1000)  # Only keep last 1000
```

### Leak 2: Callbacks and Closures

```python
class EventEmitter:
    def __init__(self):
        self.listeners = []
    
    def on(self, callback):
        self.listeners.append(callback)

emitter = EventEmitter()

def create_handler():
    large_data = [0] * 1_000_000  # 1 million integers
    
    def handler(event):
        print(f"Got event, data size: {len(large_data)}")
    
    emitter.on(handler)  # Closure captures large_data
    # large_data can never be freed while emitter exists!

create_handler()  # large_data is "leaked"
```

Fix: Use weak references for callbacks or explicit unsubscribe.

### Leak 3: Class-Level Caches

```python
class Model:
    _cache = {}  # Class variable - shared by all instances
    
    def __init__(self, data):
        # Cache grows forever as we create instances
        self._cache[id(self)] = data
    
    # No cleanup when instance is deleted!

# Fix: Use WeakValueDictionary or clean up in __del__
```

### Leak 4: Circular References in Frameworks

```python
# This pattern is common and creates cycles
class View:
    def __init__(self, controller):
        self.controller = controller
    
    def on_click(self):
        self.controller.handle_click()

class Controller:
    def __init__(self):
        self.view = View(self)  # Cycle: controller <-> view

# Each controller-view pair forms a cycle
# They'll be collected eventually by GC, but may accumulate
```

---

## 8. Memory Profiling

When we need to find memory issues, Python provides tools.

### tracemalloc: Built-in Memory Tracing

```python
import tracemalloc

# Start tracing memory allocations
tracemalloc.start()

# ... your code here ...
data = [list(range(10000)) for _ in range(100)]

# Take a snapshot
snapshot = tracemalloc.take_snapshot()

# Get top memory consumers
top_stats = snapshot.statistics('lineno')

print("Top 10 memory allocations:")
for stat in top_stats[:10]:
    print(stat)
```

Output shows which lines allocated the most memory:
```
/path/to/script.py:7: size=7648 KiB, count=100, average=76 KiB
...
```

### Comparing Snapshots

```python
import tracemalloc

tracemalloc.start()

# Take baseline snapshot
snapshot1 = tracemalloc.take_snapshot()

# Do some work
process_data()

# Take another snapshot
snapshot2 = tracemalloc.take_snapshot()

# Compare
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("Memory differences:")
for stat in top_stats[:10]:
    print(stat)
```

This shows what allocations happened between snapshots—perfect for finding leaks.

### sys.getsizeof: Object Size

```python
import sys

print(sys.getsizeof([]))        # 56 bytes (empty list)
print(sys.getsizeof([1, 2, 3])) # 120 bytes
print(sys.getsizeof("hello"))   # 54 bytes
print(sys.getsizeof(42))        # 28 bytes
```

Note: `getsizeof` returns the size of the object itself, not objects it references:

```python
big_list = [[0] * 1000 for _ in range(1000)]
print(sys.getsizeof(big_list))  # Only ~8KB - size of list of references
# Actual memory is much larger (the nested lists)
```

### Deep Size with objgraph

For deep size (including referenced objects), use third-party tools:

```python
# pip install objgraph pympler
from pympler import asizeof

big_list = [[0] * 1000 for _ in range(1000)]
print(asizeof.asizeof(big_list))  # ~32MB - actual total size
```

### Finding Object References

```python
import gc
import objgraph

class LeakyClass:
    pass

obj = LeakyClass()
some_list = [obj]
some_dict = {"key": obj}

# Find all references to obj
objgraph.show_backrefs([obj], max_depth=3, filename='refs.png')

# Or as text
for ref in gc.get_referrers(obj):
    print(type(ref), ref[:50] if isinstance(ref, (list, dict)) else ref)
```

---

## 9. Practical Debugging Workflow

When memory usage grows unexpectedly, follow this process.

### Step 1: Confirm There Is a Leak

```python
import tracemalloc
import gc

tracemalloc.start()

for i in range(10):
    process_batch()
    gc.collect()  # Force collection
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Iteration {i}: Current = {current / 1024 / 1024:.1f} MB")

# If "Current" keeps growing, we have a leak
```

### Step 2: Find What Is Growing

```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

# Run code that leaks
for _ in range(100):
    process_request()

snapshot2 = tracemalloc.take_snapshot()

# See what grew
for stat in snapshot2.compare_to(snapshot1, 'lineno')[:10]:
    print(stat)
```

### Step 3: Find Why Objects Are Not Freed

```python
import gc
import objgraph

# After running leaky code
gc.collect()

# Find objects that should have been freed
objgraph.show_growth(limit=10)

# If we know the class that's leaking:
objgraph.show_backrefs(
    objgraph.by_type('LeakyClass')[:1],
    max_depth=5,
    filename='leak.png'
)
```

### Step 4: Common Fixes

1. **Growing collections**: Add size limits or periodic cleanup
2. **Closures capturing data**: Avoid capturing large objects
3. **Callbacks not removed**: Use weak references or explicit unsubscribe
4. **Circular references**: Break cycles with weak refs or explicit cleanup

---

## 10. Memory Optimization Techniques

When memory is tight, we can optimize.

### Use Generators Instead of Lists

```python
# Uses memory for all items
data = [process(x) for x in range(1_000_000)]

# Uses memory for one item at a time
data = (process(x) for x in range(1_000_000))
```

### Use __slots__ for Many Small Objects

```python
# Normal class: each instance has a __dict__ (~100+ bytes)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# With slots: fixed attributes, no __dict__ (~50 bytes)
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

See the slots document for details.

### Use memoryview for Large Data

```python
# Copies data
data = large_bytes[1000:2000]

# Views data without copying
data = memoryview(large_bytes)[1000:2000]
```

### Use del to Release Early

```python
def process():
    large_data = load_huge_file()
    result = compute(large_data)
    
    # We don't need large_data anymore
    del large_data  # Free memory now, don't wait for function return
    
    # Continue with just result
    return format_output(result)
```

### Use Streaming for Large Data

```python
# Bad: Load entire file into memory
with open('huge.csv') as f:
    data = f.readlines()  # All in memory

# Good: Stream line by line
with open('huge.csv') as f:
    for line in f:  # One line at a time
        process(line)
```

---

## 11. Memory in Long-Running Servers

Servers have special memory concerns because they run indefinitely.

### Request-Scoped Objects

```python
# Bad: Accumulates across requests
global_cache = {}

def handle_request(request):
    key = request.id
    global_cache[key] = expensive_computation()  # Never cleared!
    return global_cache[key]

# Good: Clear after use
def handle_request(request):
    result = expensive_computation()
    return result  # No accumulation
```

### Bounded Caches

```python
from functools import lru_cache

# LRU cache with max size
@lru_cache(maxsize=1000)
def cached_computation(key):
    return expensive_computation(key)

# For more control, use cachetools
from cachetools import TTLCache, LRUCache

# Cache with time-based expiration
cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL

# Cache with size limit
cache = LRUCache(maxsize=1000)
```

### Periodic Cleanup

```python
import gc
import threading
import time

def periodic_cleanup():
    while True:
        time.sleep(300)  # Every 5 minutes
        gc.collect()  # Force garbage collection
        # Could also clear caches, trim data structures, etc.

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()
```

### Monitoring Memory

```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Log periodically
print(f"Memory usage: {get_memory_usage():.1f} MB")
```

Set up alerts if memory exceeds thresholds.

---

## 12. Interview Perspective

Common interview questions about Python memory:

**Q: How does Python manage memory?**

A: Python uses two mechanisms. Reference counting is the primary mechanism—each object tracks how many references point to it, and when the count hits zero, the object is immediately freed. For circular references (A references B, B references A), reference counting fails, so Python has a garbage collector that periodically scans for unreachable cycles.

**Q: What is a memory leak in Python?**

A: A memory leak occurs when objects remain allocated even though the program no longer needs them. Common causes include: growing collections that are never cleared, closures capturing large objects, callbacks not being unsubscribed, and circular references (though the GC handles most of these).

**Q: When is `__del__` called?**

A: `__del__` is called when an object's reference count hits zero. But timing is unpredictable with circular references (the GC decides when to collect cycles). For reliable cleanup, use context managers (`with` statement) instead of `__del__`.

**Q: What are weak references?**

A: Weak references let us reference an object without preventing its garbage collection. They are useful for caches (we cache a value but let it be freed if memory is needed) and for avoiding circular references (instead of A strongly referencing B, A weakly references B).

---

## Summary

Python's memory management combines reference counting for immediate cleanup with garbage collection for cycle detection. Understanding this helps us write memory-efficient code and debug leaks.

Key points:
- Reference counting: immediate deallocation when count hits zero
- Garbage collection: handles circular references
- `__del__` timing is unpredictable—use context managers for cleanup
- Weak references avoid keeping objects alive unnecessarily
- Use `tracemalloc` to find memory issues
- Common leaks: growing collections, captured closures, lingering callbacks

For LLM applications, memory matters when:
- Caching embeddings and model outputs
- Processing large documents
- Running long-lived servers that handle many requests
- Streaming large responses

The tools exist to find and fix memory issues. The first step is understanding how Python manages memory under the hood.
