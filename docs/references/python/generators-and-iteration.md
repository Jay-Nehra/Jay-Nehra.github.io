# Generators and Iteration

This document explains how generators and iteration work in Python—the mechanics of lazy evaluation, the iterator protocol, and practical patterns for processing data efficiently.

By the end of this guide, we will understand when to use generators over lists, how to write our own iterators, and how to build data processing pipelines that handle large datasets without loading everything into memory.

---

## 1. The Problem Generators Solve

Consider processing a large log file:

```python
# Approach 1: Load everything into memory
with open("huge_log.txt") as f:
    lines = f.readlines()  # All 10GB in memory!

for line in lines:
    process(line)
```

If the file is 10GB, we need 10GB of memory. For larger files, we crash.

```python
# Approach 2: Process one line at a time
with open("huge_log.txt") as f:
    for line in f:  # One line at a time
        process(line)
```

The second approach uses almost no memory. How? The file object is an iterator—it produces lines one at a time, on demand.

Generators let us create our own iterators that produce values lazily, without storing everything in memory.

---

## 2. The Iterator Protocol

Before generators, we need to understand what iteration actually is.

### The Two Methods

An iterator is any object that implements two methods:

- `__iter__()`: Returns the iterator object itself
- `__next__()`: Returns the next value, or raises `StopIteration` when done

```python
class CountUp:
    """Iterator that counts from start to end."""
    
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Use it
counter = CountUp(1, 4)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3
print(next(counter))  # Raises StopIteration

# Or in a for loop (which handles StopIteration)
for n in CountUp(1, 4):
    print(n)  # 1, 2, 3
```

### Iterable vs Iterator

An **iterable** is anything we can loop over. It has an `__iter__` method that returns an iterator.

An **iterator** is an object that produces values. It has `__iter__` (returns self) and `__next__` (returns next value).

```python
my_list = [1, 2, 3]  # Iterable, not an iterator

iterator = iter(my_list)  # Get an iterator from the iterable
print(next(iterator))  # 1
print(next(iterator))  # 2

# Lists are iterable but not iterators
# print(next(my_list))  # TypeError: 'list' object is not an iterator
```

### Why the Distinction Matters

Iterables can be iterated multiple times:

```python
my_list = [1, 2, 3]

for x in my_list:
    print(x)  # 1, 2, 3

for x in my_list:
    print(x)  # 1, 2, 3 again (fresh iterator each time)
```

Iterators are exhausted after one pass:

```python
iterator = iter([1, 2, 3])

for x in iterator:
    print(x)  # 1, 2, 3

for x in iterator:
    print(x)  # Nothing! Iterator is exhausted
```

---

## 3. Generator Functions

Writing iterator classes is verbose. Generators provide a simpler syntax using the `yield` keyword.

### Basic Generator Function

```python
def count_up(start, end):
    """Generator that counts from start to end."""
    current = start
    while current < end:
        yield current  # Pause here, return value
        current += 1

# Use it
for n in count_up(1, 4):
    print(n)  # 1, 2, 3
```

When we call `count_up(1, 4)`, Python does not run the function. Instead, it returns a generator object. The function body only runs when we iterate.

### How yield Works

```python
def simple_generator():
    print("First")
    yield 1
    print("Second")
    yield 2
    print("Third")
    yield 3
    print("Done")

gen = simple_generator()
print("Created generator")

print(next(gen))  # Prints "First", then yields 1
print(next(gen))  # Prints "Second", then yields 2
print(next(gen))  # Prints "Third", then yields 3
print(next(gen))  # Prints "Done", then raises StopIteration
```

Each `yield` pauses the function and returns a value. The function state (local variables, execution position) is preserved. The next `next()` call resumes from where it left off.

---

## 4. Generator Expressions

For simple cases, we can use generator expressions—like list comprehensions but with parentheses instead of brackets.

```python
# List comprehension - creates entire list in memory
squares_list = [x**2 for x in range(1_000_000)]  # ~8MB

# Generator expression - creates values on demand
squares_gen = (x**2 for x in range(1_000_000))  # ~100 bytes
```

Generator expressions are memory-efficient but can only be iterated once.

### When to Use Which

```python
# Use list comprehension when:
# - You need random access (data[500])
# - You need to iterate multiple times
# - The data fits comfortably in memory

squares = [x**2 for x in range(100)]
print(squares[50])  # Random access
print(len(squares))  # Length

# Use generator expression when:
# - You only need to iterate once
# - Data is large or unbounded
# - You're feeding directly into another function

total = sum(x**2 for x in range(1_000_000))  # Memory-efficient
```

---

## 5. yield from: Delegating to Sub-Generators

When a generator needs to yield values from another iterable, use `yield from`:

```python
# Without yield from
def chain_manually(*iterables):
    for iterable in iterables:
        for item in iterable:
            yield item

# With yield from (cleaner)
def chain(*iterables):
    for iterable in iterables:
        yield from iterable

# Use it
for x in chain([1, 2], [3, 4], [5, 6]):
    print(x)  # 1, 2, 3, 4, 5, 6
```

---

## 6. Practical Patterns

### Pattern 1: Processing Large Files

```python
def process_log_file(filepath):
    """Process a large log file without loading it entirely."""
    with open(filepath) as f:
        for line in f:  # File objects are iterators
            if "ERROR" in line:
                yield parse_error(line)

for error in process_log_file("huge.log"):
    handle_error(error)
```

### Pattern 2: Chunking Documents

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for embedding."""
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start = end - overlap

document = load_document()
for chunk in chunk_text(document):
    embedding = generate_embedding(chunk)
    store(embedding)
```

### Pattern 3: Pipeline of Transformations

```python
def read_lines(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()

def filter_non_empty(lines):
    for line in lines:
        if line:
            yield line

def parse_json(lines):
    import json
    for line in lines:
        yield json.loads(line)

# Build pipeline - nothing happens until we iterate
records = parse_json(filter_non_empty(read_lines("data.jsonl")))

for record in records:
    process(record)
```

---

## 7. The itertools Module

The `itertools` module provides efficient building blocks:

```python
from itertools import chain, islice, groupby

# chain: concatenate iterables
for x in chain([1, 2], [3, 4]):
    print(x)  # 1, 2, 3, 4

# islice: slice an iterator
from itertools import count
list(islice(count(), 5, 10))  # [5, 6, 7, 8, 9]

# Batching (Python 3.12+ has itertools.batched)
def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

list(batched(range(10), 3))  # [(0,1,2), (3,4,5), (6,7,8), (9,)]
```

---

## 8. Common Mistakes

### Mistake 1: Trying to Iterate Twice

```python
gen = (x**2 for x in range(5))
print(list(gen))  # [0, 1, 4, 9, 16]
print(list(gen))  # [] - exhausted!

# Fix: recreate the generator
def make_gen():
    return (x**2 for x in range(5))
```

### Mistake 2: Checking Length

```python
gen = (x for x in range(10))
# len(gen)  # TypeError

# Fix: convert to list or count manually
count = sum(1 for _ in gen)  # But this exhausts it!
```

---

## Summary

Generators provide lazy evaluation—computing values on demand. Use them when:

- Processing data larger than memory
- Streaming responses
- Building data pipelines

Key concepts:
- Iterators have `__iter__` and `__next__`
- Generators use `yield` to create iterators simply
- Generator expressions are lazy list comprehensions
- Generators exhaust after one pass
