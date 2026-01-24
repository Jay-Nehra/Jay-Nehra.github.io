# The Python Import System

This document explains how Python's import system works—what happens when we write `import module`, where Python looks for modules, and how to structure projects to avoid import problems.

---

## 1. What Happens When We Import

When we write `import mymodule`, Python performs:

```
1. Check cache (sys.modules)
   └── Found? → Return cached module
   └── Not found? → Continue

2. Find the module
   └── Search sys.path for mymodule.py or mymodule/

3. Create empty module object
   └── Add to sys.modules (before executing!)

4. Execute the module code
   └── All top-level code runs

5. Bind the name
   └── "mymodule" now refers to the module object
```

### The Cache

```python
import sys

# First import: loads and executes
import mymodule

# Second import: returns cached, no re-execution
import mymodule

print('mymodule' in sys.modules)  # True
```

### Where Python Looks

```python
import sys
print(sys.path)
# ['', '/usr/lib/python3.11', ...]
```

The empty string `''` means current directory.

---

## 2. Import Statements

### import module

```python
import os
print(os.path.exists("/tmp"))
```

### from module import name

```python
from os.path import exists
print(exists("/tmp"))
```

### import as

```python
import numpy as np
import pandas as pd
```

---

## 3. Packages

A package is a directory with `__init__.py`:

```
mypackage/
    __init__.py
    module_a.py
    subpackage/
        __init__.py
        module_b.py
```

When importing a package, `__init__.py` executes.

---

## 4. Absolute vs Relative Imports

### Absolute Imports

```python
# mypackage/module_a.py
from mypackage.module_b import something
```

### Relative Imports

```python
# mypackage/module_a.py
from .module_b import something       # Same directory
from ..other_package import thing     # Parent directory
```

Relative imports only work inside packages, not scripts.

---

## 5. Circular Imports

When module A imports B and B imports A:

```python
# module_a.py
from module_b import func_b  # B not finished loading!

# module_b.py
from module_a import func_a  # A not finished loading!
```

### Solutions

**Import at function level:**
```python
def my_function():
    from module_b import func_b
    return func_b()
```

**Import the module, not the name:**
```python
import module_b

def my_function():
    return module_b.func_b()
```

**Restructure to avoid cycles.**

---

## 6. Lazy Imports

```python
_numpy = None

def get_numpy():
    global _numpy
    if _numpy is None:
        import numpy
        _numpy = numpy
    return _numpy
```

---

## 7. The `if __name__ == "__main__"` Pattern

```python
def main():
    print("Running as script")

if __name__ == "__main__":
    main()  # Only runs when executed directly
```

---

## 8. Common Errors

### ModuleNotFoundError

- Module not installed
- Not in `sys.path`
- Typo

### ImportError: cannot import name

- Name doesn't exist in module
- Circular import
- Typo

### Shadowing

```python
# random.py (your file)
import random  # Imports YOUR file, not stdlib!
```

Don't name files after stdlib modules.

---

## Summary

- Imports are cached in `sys.modules`
- Top-level code runs on import
- Use relative imports within packages
- Avoid circular imports through restructuring
- Use `if __name__ == "__main__"` for runnable modules
