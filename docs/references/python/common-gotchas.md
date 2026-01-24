# Common Python Gotchas

This document covers the classic Python pitfalls that trip up developers in interviews and production code. For each gotcha, we explain what happens, why it happens, and how to fix it.

These are not obscure corner cases—they appear regularly in real code and are favorite interview questions because they test understanding of Python's execution model.

---

## 1. Mutable Default Arguments

This is the most famous Python gotcha.

### The Problem

```python
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] - Wait, what?!
print(add_item("c"))  # ['a', 'b', 'c'] - The list keeps growing!
```

We expected three separate lists, but got the same list accumulating items.

### Why It Happens

Default arguments are evaluated **once**, at function definition time, not at call time. The empty list `[]` is created when Python parses the `def` statement, and the same list object is reused for every call.

```python
def add_item(item, items=[]):
    print(f"items id: {id(items)}")  # Same id every call!
    items.append(item)
    return items
```

### The Fix

Use `None` as the default and create the mutable object inside the function:

```python
def add_item(item, items=None):
    if items is None:
        items = []  # Fresh list for each call
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['b'] - Fresh list!
```

### Where This Bites in Practice

```python
# Building up conversation history - BUG!
def chat(message, history=[]):
    history.append({"user": message})
    response = get_llm_response(history)
    history.append({"assistant": response})
    return response

# All users share the same history!
```

---

## 2. is vs == (Identity vs Equality)

### The Problem

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)  # True - same value
print(a is b)  # False - different objects
```

This is clear. But then:

```python
a = 256
b = 256
print(a is b)  # True

a = 257
b = 257
print(a is b)  # False - Wait, why?!
```

### Why It Happens

`==` compares **values** (calls `__eq__`).
`is` compares **identity** (same object in memory).

For small integers (-5 to 256), Python caches them for efficiency. When we write `256`, Python returns the cached object. When we write `257`, Python creates a new object.

```python
# Same object (cached)
a = 256
b = 256
print(id(a), id(b))  # Same id

# Different objects (not cached)
a = 257
b = 257
print(id(a), id(b))  # Different ids
```

### The Rule

**Always use `==` for value comparison.**

**Only use `is` for:**
- `is None`
- `is True` / `is False`
- Checking if two variables point to the same object

```python
# Correct
if x is None:
    pass

# Incorrect - could fail for large numbers
if x is 1000:
    pass

# Correct
if x == 1000:
    pass
```

### String Interning Surprise

Strings can also be interned (cached):

```python
a = "hello"
b = "hello"
print(a is b)  # True (interned)

a = "hello world!"
b = "hello world!"
print(a is b)  # False (not interned - has special chars)
```

Do not rely on this. Always use `==` for string comparison.

---

## 3. Late Binding Closures

### The Problem

```python
functions = []
for i in range(3):
    functions.append(lambda: i)

print([f() for f in functions])  # [2, 2, 2] - Not [0, 1, 2]!
```

All three lambdas return `2`, not their respective values.

### Why It Happens

Closures capture **variables**, not **values**. The variable `i` is captured by reference. When the lambdas are finally called, the loop has finished and `i` is `2`.

```python
# At definition time: lambda captures the variable i
# At call time: lambda looks up current value of i

for i in range(3):
    functions.append(lambda: i)  # Captures variable i, not value

# After loop: i = 2
# All lambdas look up i, find 2
```

### The Fix

Capture the value by making it a default argument:

```python
functions = []
for i in range(3):
    functions.append(lambda i=i: i)  # i=i captures current value

print([f() for f in functions])  # [0, 1, 2]
```

Or use `functools.partial`:

```python
from functools import partial

functions = []
for i in range(3):
    functions.append(partial(lambda x: x, i))

print([f() for f in functions])  # [0, 1, 2]
```

### Where This Bites in Practice

```python
# Creating callbacks in a loop - BUG!
buttons = []
for i, name in enumerate(["Save", "Load", "Quit"]):
    button = Button(name, onclick=lambda: handle_click(i))
    buttons.append(button)

# All buttons call handle_click(2)!
```

---

## 4. Modifying a List While Iterating

### The Problem

```python
numbers = [1, 2, 3, 4, 5]
for n in numbers:
    if n % 2 == 0:
        numbers.remove(n)

print(numbers)  # [1, 3, 5]? No! [1, 3, 5] - Actually this works...

# But try this:
numbers = [1, 2, 2, 3, 4, 5]
for n in numbers:
    if n == 2:
        numbers.remove(n)

print(numbers)  # [1, 2, 3, 4, 5] - One 2 is still there!
```

### Why It Happens

When we remove an element, all subsequent elements shift left. The iterator's internal index does not adjust.

```python
# numbers = [1, 2, 2, 3, 4, 5]
# Index:     0  1  2  3  4  5

# Iteration 1: index 0, sees 1, keeps it
# Iteration 2: index 1, sees 2, removes it
#   Now: [1, 2, 3, 4, 5]
#   Index: 0  1  2  3  4
# Iteration 3: index 2, sees 3 (skipped the second 2!)
```

### The Fix

**Option 1: Iterate over a copy**

```python
for n in numbers[:]:  # Slice creates a copy
    if n == 2:
        numbers.remove(n)
```

**Option 2: Build a new list**

```python
numbers = [n for n in numbers if n != 2]
```

**Option 3: Iterate backwards**

```python
for i in range(len(numbers) - 1, -1, -1):
    if numbers[i] == 2:
        del numbers[i]
```

### Also Applies to Dicts

```python
# BAD - RuntimeError: dictionary changed size during iteration
for key in my_dict:
    if should_remove(key):
        del my_dict[key]

# GOOD
keys_to_remove = [k for k in my_dict if should_remove(k)]
for key in keys_to_remove:
    del my_dict[key]

# Or use dict comprehension
my_dict = {k: v for k, v in my_dict.items() if not should_remove(k)}
```

---

## 5. UnboundLocalError: Variable Scope Surprise

### The Problem

```python
x = 10

def foo():
    print(x)  # UnboundLocalError: local variable 'x' referenced before assignment
    x = 20

foo()
```

Why does this fail? We are just reading `x` before assigning to it.

### Why It Happens

Python determines variable scope at **compile time**, not runtime. When Python sees `x = 20` anywhere in the function, it marks `x` as local for the **entire** function.

```python
def foo():
    print(x)  # x is local (because of line below), but not yet assigned
    x = 20    # This makes x local to the whole function
```

### The Fix

**If you want to read the global:**

```python
x = 10

def foo():
    print(x)  # Works - no local x

foo()  # 10
```

**If you want to modify the global:**

```python
x = 10

def foo():
    global x
    print(x)  # 10
    x = 20

foo()
print(x)  # 20
```

**If you want both local and global:**

```python
x = 10

def foo():
    local_x = x  # Read global first
    local_x = 20  # Then work with local
```

### Related: nonlocal for Closures

```python
def outer():
    x = 10
    
    def inner():
        nonlocal x  # Needed to modify x from outer scope
        x = 20
    
    inner()
    print(x)  # 20

outer()
```

---

## 6. Tuple Unpacking Edge Cases

### Single-Element Tuple

```python
t = (1)    # This is just the integer 1
t = (1,)   # This is a tuple with one element

print(type((1)))   # <class 'int'>
print(type((1,)))  # <class 'tuple'>
```

The trailing comma makes it a tuple.

### Accidental Tuple Creation

```python
x = 1, 2, 3  # This is a tuple!
print(type(x))  # <class 'tuple'>

# Common mistake in returns:
def get_coords():
    return 1, 2  # Returns tuple (1, 2), not two values

x = get_coords()  # x is (1, 2)
```

### Unpacking Mismatch

```python
a, b = [1, 2, 3]  # ValueError: too many values to unpack

# Use extended unpacking
a, *b = [1, 2, 3]  # a=1, b=[2, 3]
a, *b, c = [1, 2, 3, 4]  # a=1, b=[2, 3], c=4
```

---

## 7. Class Variable vs Instance Variable

### The Problem

```python
class User:
    roles = []  # Class variable
    
    def add_role(self, role):
        self.roles.append(role)

alice = User()
bob = User()

alice.add_role("admin")
print(bob.roles)  # ['admin'] - Wait, Bob is admin too?!
```

### Why It Happens

`roles = []` is a **class variable**—shared by all instances. When we modify it via `self.roles.append()`, we are modifying the shared list.

```python
print(alice.roles is bob.roles)  # True - same list!
print(alice.roles is User.roles)  # True
```

### The Fix

Initialize mutable objects in `__init__`:

```python
class User:
    def __init__(self):
        self.roles = []  # Instance variable
    
    def add_role(self, role):
        self.roles.append(role)

alice = User()
bob = User()
alice.add_role("admin")
print(bob.roles)  # [] - Bob has his own list
```

### When Class Variables Are OK

For immutable values that are truly shared:

```python
class Config:
    DEBUG = False  # OK - immutable
    VERSION = "1.0"  # OK - immutable
    
    # Still problematic if mutated:
    ALLOWED_HOSTS = []  # Shared mutable - dangerous!
```

---

## 8. Exception Handling: else and finally

### The else Clause

```python
try:
    result = do_something()
except ValueError:
    print("Error!")
else:
    print("Success!")  # Only runs if NO exception
finally:
    print("Always runs")
```

The `else` only runs if the `try` block completes without exception. Many people do not know this exists.

### Why else Is Useful

```python
# Without else - more in try block than necessary
try:
    data = fetch_data()
    process(data)  # If this fails, we catch the wrong error
except NetworkError:
    handle_error()

# With else - cleaner separation
try:
    data = fetch_data()
except NetworkError:
    handle_error()
else:
    process(data)  # Not inside try, so its exceptions aren't caught
```

### finally Gotcha: Return Override

```python
def example():
    try:
        return "try"
    finally:
        return "finally"

print(example())  # "finally" - finally's return wins!
```

`finally` always runs, even if `try` returned. If `finally` also returns, it overrides the `try` return.

---

## 9. Empty Collections Are Falsy

### The Behavior

```python
if []:
    print("truthy")
else:
    print("falsy")  # This prints

# These are all falsy:
bool([])      # False
bool({})      # False
bool(set())   # False
bool("")      # False
bool(0)       # False
bool(None)    # False
```

### The Gotcha

```python
def process(items=None):
    if not items:  # BUG: treats empty list same as None
        items = get_default_items()
    return items

process([])  # Returns get_default_items(), not empty list!
```

### The Fix

Be explicit about what we are checking:

```python
def process(items=None):
    if items is None:  # Only None, not empty list
        items = get_default_items()
    return items

process([])  # Returns [], as expected
```

---

## 10. Assignment Expressions (:=) Scope

### The Walrus Operator

```python
# Without walrus
match = pattern.search(text)
if match:
    print(match.group())

# With walrus
if match := pattern.search(text):
    print(match.group())
```

### The Scope Gotcha

```python
# In comprehensions, := leaks to outer scope
[y := x * 2 for x in range(3)]
print(y)  # 4 - y exists outside the comprehension!

# Compare to normal comprehension variable
[x * 2 for x in range(3)]
print(x)  # NameError - x doesn't exist
```

This is intentional behavior (to make the walrus operator useful), but can be surprising.

---

## 11. Chained Comparisons

### The Feature

```python
x = 5
if 0 < x < 10:  # Works as expected!
    print("in range")
```

This is equivalent to `0 < x and x < 10`.

### The Gotcha

```python
# This is valid but confusing:
print(1 < 2 > 1.5)  # True (1 < 2 and 2 > 1.5)

# This might surprise:
print(1 == 1 in [1, 2])  # True
# Parsed as: 1 == 1 and 1 in [1, 2]
```

### Best Practice

Only use chained comparisons for obvious ranges:

```python
# Clear
if 0 <= index < len(items):
    pass

# Confusing - don't do this
if a < b > c < d:
    pass
```

---

## 12. Creating Copies of Objects

### The Shallow vs Deep Problem

```python
import copy

original = [[1, 2], [3, 4]]

shallow = original.copy()  # or original[:]
deep = copy.deepcopy(original)

original[0][0] = 999

print(shallow)  # [[999, 2], [3, 4]] - Inner list changed!
print(deep)     # [[1, 2], [3, 4]] - Independent copy
```

### Why It Happens

`copy()` creates a new list but does not copy the elements. The inner lists are the same objects.

```python
print(original[0] is shallow[0])  # True - same inner list
print(original[0] is deep[0])     # False - different inner list
```

### When This Matters

```python
# Copying a dict of lists
config = {"handlers": [handler1, handler2]}
new_config = config.copy()
new_config["handlers"].append(handler3)
print(config["handlers"])  # [handler1, handler2, handler3] - Modified!
```

### The Fix

Use `copy.deepcopy` when you have nested mutable objects:

```python
import copy
new_config = copy.deepcopy(config)
```

---

## 13. Floating Point Precision

### The Problem

```python
print(0.1 + 0.2)  # 0.30000000000000004
print(0.1 + 0.2 == 0.3)  # False!
```

### Why It Happens

Floating point numbers cannot exactly represent most decimal fractions. `0.1` is actually `0.1000000000000000055511151231257827021181583404541015625`.

### The Fix

**For money/exact decimals: use Decimal**

```python
from decimal import Decimal

print(Decimal("0.1") + Decimal("0.2"))  # 0.3
print(Decimal("0.1") + Decimal("0.2") == Decimal("0.3"))  # True
```

**For approximate comparisons: use tolerance**

```python
import math

a = 0.1 + 0.2
b = 0.3
print(math.isclose(a, b))  # True
```

---

## Summary Table

| Gotcha | Symptom | Fix |
|--------|---------|-----|
| Mutable default | Function "remembers" previous calls | Use `None` as default |
| `is` vs `==` | Comparison works sometimes, not others | Always use `==` for values |
| Late binding | All lambdas return same value | Use default argument `i=i` |
| Modifying while iterating | Items skipped or missed | Iterate over copy |
| UnboundLocalError | Can't read variable before assigning | Use `global` or restructure |
| Class vs instance variable | Changes affect all instances | Initialize in `__init__` |
| Falsy empty collections | `if not items` catches `[]` | Use `if items is None` |
| Shallow copy | Changes to nested objects propagate | Use `copy.deepcopy` |
| Floating point | `0.1 + 0.2 != 0.3` | Use `Decimal` or `math.isclose` |

---

## Interview Perspective

These gotchas are popular interview questions because they test:

1. **Understanding of evaluation model**: When are things evaluated? Definition time vs call time.
2. **Understanding of mutability**: What can change? What is shared?
3. **Understanding of scope**: Where do names live? When are they resolved?
4. **Attention to detail**: Can you spot subtle bugs?

When answering, explain **why** the behavior occurs, not just what the fix is. That demonstrates real understanding.
