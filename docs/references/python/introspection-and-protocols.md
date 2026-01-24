# Introspection and Python Protocols

This document explains how to use Python's introspection tools—especially `dir()`—to understand what any object can do without reading documentation. We will learn to read the "capability matrix" that every Python object carries with it.

More importantly, we will understand **protocols**: the combinations of dunder methods that give objects specific behaviors. If an object has `__iter__` and `__next__`, it is an iterator. If it has `__enter__` and `__exit__`, it is a context manager. By recognizing these patterns, we can understand any object instantly.

---

## 1. The Philosophy: Objects Describe Themselves

In Python, every object carries a manifest of its own capabilities. Unlike languages where we must read class definitions or documentation, Python lets us ask the object directly: "What can you do?"

The primary tool for this interrogation is `dir()`. Its output might look like noise at first—a wall of strings with strange punctuation—but it is actually a complete map of the object's potential.

```python
>>> dir([1, 2, 3])
['__add__', '__class__', '__contains__', '__delattr__', '__delitem__',
 '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
 '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__',
 '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__',
 '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__',
 '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__',
 '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index',
 'insert', 'pop', 'remove', 'reverse', 'sort']
```

This is not random. Every name tells us something specific about what this list can do.

---

## 2. How dir() Works

Understanding the mechanics helps us interpret the output.

### Without Arguments: Local Scope

```python
>>> x = 10
>>> y = "hello"
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
 '__package__', '__spec__', 'x', 'y']
```

Without arguments, `dir()` shows what names are defined in the current scope. This answers: "What variables do I have right now?"

### With an Object: Its Namespace

```python
>>> dir("hello")
['__add__', '__class__', ..., 'upper', 'lower', 'split', ...]
```

With an object, `dir()` shows everything that object can do—methods, attributes, and the dunder hooks that connect it to Python's syntax.

### The Lookup Order

When we call `dir(obj)`, Python looks for attributes in:

1. The object's `__dict__` (instance attributes)
2. The object's class (methods and class attributes)
3. All parent classes (inherited methods)

If the class defines `__dir__()`, Python uses that instead. This means objects can customize what `dir()` returns.

---

## 3. The Three Categories in dir() Output

Every name in `dir()` output falls into one of three categories:

### Category 1: Public API (No Underscores)

Names like `append`, `split`, `read` are the **public interface**—the tools designed for us to use.

```python
>>> [name for name in dir([]) if not name.startswith('_')]
['append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 
 'pop', 'remove', 'reverse', 'sort']
```

These are safe, documented, and stable. When exploring an object, look at these first.

### Category 2: Internal/Protected (Single Underscore)

Names like `_cache`, `_internal_state` are **internal implementation details**. We can access them, but doing so bypasses safety checks.

```python
# Convention: "I'm internal, don't touch unless you know what you're doing"
obj._internal_data
```

### Category 3: Protocols/Dunders (Double Underscores)

Names like `__len__`, `__iter__`, `__add__` are the **protocol methods**—the hooks that connect the object to Python's syntax.

These are the most important for understanding what an object can do.

---

## 4. Reading Dunder Methods: The Capability Matrix

Dunder methods are not meant to be called directly. They define what operations the object supports. Their presence in `dir()` is a promise.

### The Mental Model

When we see a dunder method, we should ask: "What Python feature does this enable?"

| If you see... | The object supports... |
|---------------|------------------------|
| `__len__` | `len(obj)` |
| `__iter__` | `for x in obj:` |
| `__getitem__` | `obj[key]` |
| `__call__` | `obj()` |
| `__add__` | `obj + other` |
| `__enter__`, `__exit__` | `with obj:` |

Let us go through each protocol systematically.

---

## 5. Protocol: Sized Objects (`__len__`)

If `__len__` appears in `dir()`, the object has a finite size.

```python
>>> '__len__' in dir([1, 2, 3])
True

>>> len([1, 2, 3])
3
```

**What this enables:** The `len()` function, truthiness testing for empty containers.

**Objects with this:** Lists, strings, dicts, sets, tuples, most collections.

**Objects without this:** Generators (they do not know their length until exhausted).

---

## 6. Protocol: Iterable (`__iter__`)

If `__iter__` appears, the object can be looped over.

```python
>>> '__iter__' in dir([1, 2, 3])
True

>>> for x in [1, 2, 3]:
...     print(x)
```

**What this enables:** `for` loops, unpacking, `list()`, `sum()`, and any function that consumes iterables.

**The iterator subtype:** If an object has both `__iter__` AND `__next__`, it is an iterator—a stateful stream that exhausts after one pass.

```python
# List: iterable but not iterator
>>> '__next__' in dir([1, 2, 3])
False

# File: is an iterator
>>> f = open('test.txt')
>>> '__next__' in dir(f)
True
```

---

## 7. Protocol: Indexable (`__getitem__`)

If `__getitem__` appears, the object supports bracket access.

```python
>>> '__getitem__' in dir([1, 2, 3])
True

>>> [1, 2, 3][0]
1
```

**What this enables:** `obj[key]`, slicing `obj[1:3]`.

**Sequences vs Mappings:** Both lists and dicts have `__getitem__`, but lists use integer indices while dicts use keys. Check for `keys()` in the public API to distinguish.

**Mutability check:** If `__setitem__` is also present, we can assign: `obj[key] = value`. If only `__getitem__` exists, the object is read-only (like tuples or strings).

```python
# List: mutable
>>> '__setitem__' in dir([])
True

# Tuple: immutable
>>> '__setitem__' in dir(())
False
```

---

## 8. Protocol: Container (`__contains__`)

If `__contains__` appears, the object supports the `in` operator.

```python
>>> '__contains__' in dir([1, 2, 3])
True

>>> 2 in [1, 2, 3]
True
```

**What this enables:** `if x in obj:` membership testing.

**Fallback:** If `__contains__` is missing but `__iter__` exists, Python iterates through the object to check membership (slower).

---

## 9. Protocol: Callable (`__call__`)

If `__call__` appears, the object can be used like a function.

```python
>>> '__call__' in dir(len)
True

>>> len([1, 2, 3])  # Calling a callable
3
```

**What this enables:** `obj()` invocation with parentheses.

**Surprisingly callable:** Classes are callable (calling them creates instances). Functions are objects with `__call__`.

```python
>>> '__call__' in dir(list)
True

>>> list()  # Calling the class creates an instance
[]
```

---

## 10. Protocol: Context Manager (`__enter__` + `__exit__`)

If both `__enter__` AND `__exit__` appear, the object is a context manager.

```python
>>> f = open('test.txt', 'w')
>>> '__enter__' in dir(f) and '__exit__' in dir(f)
True
```

**What this enables:** The `with` statement for automatic resource management.

```python
with open('file.txt') as f:
    data = f.read()
# f is automatically closed, even if exception occurs
```

**The signal:** When you see these methods, use `with`. Do not manually manage the resource.

---

## 11. Protocol: Comparable (`__eq__`, `__lt__`, etc.)

Comparison methods enable relational operators:

| Dunder | Operator |
|--------|----------|
| `__eq__` | `==` |
| `__ne__` | `!=` |
| `__lt__` | `<` |
| `__le__` | `<=` |
| `__gt__` | `>` |
| `__ge__` | `>=` |

**Sorting requirement:** For an object to be sortable, it needs at least `__lt__`.

```python
>>> '__lt__' in dir(5)
True

>>> sorted([3, 1, 2])
[1, 2, 3]
```

---

## 12. Protocol: Hashable (`__hash__` + `__eq__`)

If `__hash__` appears (and is not `None`), the object can be used as a dictionary key or stored in a set.

```python
>>> '__hash__' in dir("hello")
True

>>> {("hello"): 1}  # String as dict key
{'hello': 1}

>>> '__hash__' in dir([1, 2, 3])
True

>>> hash([1, 2, 3])  # But this fails!
TypeError: unhashable type: 'list'
```

**The trap:** Lists have `__hash__` in `dir()`, but it is set to `None`. Check by actually calling `hash()` or checking if `obj.__hash__ is None`.

**The rule:** Mutable objects should not be hashable (because their hash would change when modified, breaking dict lookups).

---

## 13. Protocol: Arithmetic (`__add__`, `__mul__`, etc.)

Arithmetic dunders enable operators:

| Dunder | Operator | Example |
|--------|----------|---------|
| `__add__` | `+` | `obj + other` |
| `__sub__` | `-` | `obj - other` |
| `__mul__` | `*` | `obj * other` |
| `__truediv__` | `/` | `obj / other` |
| `__floordiv__` | `//` | `obj // other` |
| `__mod__` | `%` | `obj % other` |
| `__pow__` | `**` | `obj ** other` |

### Reflected Operators (`__radd__`, `__rmul__`)

If you see `__radd__`, the object can appear on the right side of `+`:

```python
# When 10 + obj is called:
# 1. Python tries 10.__add__(obj)
# 2. If that fails, tries obj.__radd__(10)
```

### In-Place Operators (`__iadd__`, `__imul__`)

If you see `__iadd__`, the object supports in-place modification:

```python
obj += 1  # Calls __iadd__ if present, modifies in place
```

If `__iadd__` is missing but `__add__` exists, `obj += 1` becomes `obj = obj + 1` (creates new object).

---

## 14. Protocol: String Representation (`__str__`, `__repr__`)

These control how the object appears as text:

| Dunder | Used By | Purpose |
|--------|---------|---------|
| `__str__` | `print()`, `str()` | Human-readable output |
| `__repr__` | Interactive console, `repr()` | Developer-readable, unambiguous |

```python
>>> class Point:
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...     def __repr__(self):
...         return f"Point({self.x}, {self.y})"
...     def __str__(self):
...         return f"({self.x}, {self.y})"

>>> p = Point(1, 2)
>>> repr(p)
'Point(1, 2)'
>>> str(p)
'(1, 2)'
```

---

## 15. Quick Reference: Protocol Combinations

Here are the common protocol "signatures" to recognize:

### Sequence (like list, tuple)

```python
Required: __len__, __getitem__
Optional: __iter__, __contains__, __reversed__
Mutable: + __setitem__, __delitem__
```

### Mapping (like dict)

```python
Required: __len__, __getitem__, __iter__
Typically has: keys(), values(), items()
Mutable: + __setitem__, __delitem__
```

### Iterator

```python
Required: __iter__ (returns self), __next__
```

### Context Manager

```python
Required: __enter__, __exit__
```

### Callable

```python
Required: __call__
```

### Number-like

```python
Typically has: __add__, __sub__, __mul__, __truediv__
Plus: __neg__, __abs__, __int__, __float__
```

---

## 16. Practical Workflow: Exploring Unknown Objects

When faced with an unfamiliar object, follow this process:

### Step 1: Get the Type

```python
>>> type(mystery_obj)
<class 'pandas.core.frame.DataFrame'>
```

### Step 2: Scan Public Methods

```python
>>> [m for m in dir(mystery_obj) if not m.startswith('_')]
['abs', 'add', 'agg', 'aggregate', 'align', 'all', 'any', ...]
```

### Step 3: Check Key Protocols

```python
>>> '__iter__' in dir(mystery_obj)  # Can I loop over it?
True

>>> '__len__' in dir(mystery_obj)   # Does it have a size?
True

>>> '__getitem__' in dir(mystery_obj)  # Can I index it?
True
```

### Step 4: Get Help on Specific Methods

```python
>>> help(mystery_obj.head)
Help on method head in module pandas.core.generic:
...
```

### Step 5: Try It

```python
>>> len(mystery_obj)
100

>>> mystery_obj[0]
# See what happens
```

---

## 17. Debugging with Introspection

### "AttributeError: object has no attribute 'x'"

```python
>>> dir(obj)  # Check what attributes exist
>>> 'x' in dir(obj)  # Is 'x' there?
False

# Maybe a typo? Check similar names:
>>> [a for a in dir(obj) if 'x' in a.lower()]
```

### "TypeError: object is not iterable"

```python
>>> '__iter__' in dir(obj)
False  # Confirms: this object cannot be looped over
```

### "TypeError: object is not subscriptable"

```python
>>> '__getitem__' in dir(obj)
False  # Confirms: cannot use obj[key]
```

---

## 18. Beyond dir(): Related Tools

### vars(): Instance Attributes Only

```python
>>> class Dog:
...     def __init__(self, name):
...         self.name = name

>>> d = Dog("Rex")
>>> vars(d)
{'name': 'Rex'}

>>> dir(d)  # Includes methods, inherited stuff
['__class__', ..., 'name', ...]
```

`vars()` returns the `__dict__`—just the instance's own attributes.

### type(): What Kind of Object

```python
>>> type([1, 2, 3])
<class 'list'>

>>> type(len)
<class 'builtin_function_or_method'>
```

### isinstance(): Check Type Hierarchy

```python
>>> isinstance([1, 2, 3], list)
True

>>> isinstance([1, 2, 3], (list, tuple))  # Check multiple types
True
```

### hasattr(): Check for Specific Attribute

```python
>>> hasattr([1, 2, 3], 'append')
True

>>> hasattr([1, 2, 3], 'add')
False
```

### getattr(): Access Attribute by Name

```python
>>> getattr([1, 2, 3], 'append')
<built-in method append of list object at 0x...>

>>> method_name = 'append'
>>> getattr([1, 2, 3], method_name)([4])
```

---

## 19. The Inspection Mindset

Reading `dir()` is a skill that develops with practice. The key shifts in thinking:

1. **From "what is this?" to "what can I do with this?"**
   - Do not just check the type. Check the capabilities.

2. **From memorizing APIs to recognizing patterns**
   - Once you know `__iter__` means "iterable," you recognize it everywhere.

3. **From reading documentation to interrogating objects**
   - The object tells you what it can do. Trust it.

4. **From trial-and-error to informed experimentation**
   - Check `dir()` first, then try operations you know are supported.

---

## Summary

`dir()` is not a wall of noise—it is a capability matrix. Every name has meaning:

- **No underscores**: Public API, safe to use
- **Single underscore**: Internal, use with caution
- **Double underscore (dunder)**: Protocol method, defines what operations are supported

Key protocols to recognize:

| Protocol | Required Dunders | Enables |
|----------|------------------|---------|
| Sized | `__len__` | `len(obj)` |
| Iterable | `__iter__` | `for x in obj` |
| Iterator | `__iter__` + `__next__` | Stateful iteration |
| Sequence | `__len__` + `__getitem__` | `obj[i]`, slicing |
| Container | `__contains__` | `x in obj` |
| Callable | `__call__` | `obj()` |
| Context Manager | `__enter__` + `__exit__` | `with obj:` |
| Hashable | `__hash__` + `__eq__` | Dict keys, sets |
| Comparable | `__eq__`, `__lt__`, etc. | `==`, `<`, sorting |

By learning to read these patterns, we move from being consumers of documentation to active interrogators of our runtime environment. Any object, no matter how unfamiliar, reveals its nature through `dir()`.
