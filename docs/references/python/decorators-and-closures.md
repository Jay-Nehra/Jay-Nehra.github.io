# Decorators and Closures

This document explains how decorators and closures work in Python—not just the syntax, but the underlying mechanics. We will understand why decorators look the way they do, how closures capture variables, and how to write practical decorators for real applications.

By the end of this guide, we will know how to write decorators with and without arguments, debug decorator issues, and apply common patterns like retry, caching, and timing.

---

## 1. Functions Are Objects

Before we can understand decorators, we need to internalize that functions in Python are objects. They can be assigned to variables, passed as arguments, and returned from other functions.

### Functions as Variables

```python
def greet(name):
    return f"Hello, {name}"

# greet is just a variable pointing to a function object
print(type(greet))  # <class 'function'>

# We can assign it to another variable
say_hello = greet
print(say_hello("Alice"))  # Hello, Alice

# We can put it in a list
functions = [greet, len, print]
print(functions[0]("Bob"))  # Hello, Bob
```

### Functions as Arguments

```python
def apply_twice(func, value):
    return func(func(value))

def add_one(x):
    return x + 1

result = apply_twice(add_one, 5)
print(result)  # 7 (5 -> 6 -> 7)
```

We passed `add_one` as an argument to `apply_twice`. The function is just data.

### Functions Returning Functions

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

`make_multiplier` returns a function. Each returned function is different—one doubles, one triples.

---

## 2. What Is a Closure

A closure is a function that "remembers" variables from its enclosing scope, even after that scope has finished executing.

### The Basic Mechanism

```python
def make_counter():
    count = 0
    
    def counter():
        nonlocal count
        count += 1
        return count
    
    return counter

my_counter = make_counter()
print(my_counter())  # 1
print(my_counter())  # 2
print(my_counter())  # 3
```

When `make_counter()` returns, its local variable `count` should disappear. But the inner function `counter` still references it. Python keeps `count` alive because `counter` needs it.

This is a closure: `counter` "closes over" the variable `count`.

### Viewing Closure Variables

```python
def make_greeter(greeting):
    def greeter(name):
        return f"{greeting}, {name}!"
    return greeter

say_hello = make_greeter("Hello")
say_hi = make_greeter("Hi")

# We can inspect what the closure captured
print(say_hello.__closure__)  # (<cell at 0x...>,)
print(say_hello.__closure__[0].cell_contents)  # "Hello"
print(say_hi.__closure__[0].cell_contents)  # "Hi"
```

Each function has its own `__closure__` containing the captured values.

### The LEGB Rule

Python looks up variables in this order:
- **L**ocal: Inside the current function
- **E**nclosing: In enclosing function scopes (closures)
- **G**lobal: At the module level
- **B**uilt-in: Python's built-in names

```python
x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # local
    
    inner()
    print(x)  # enclosing

outer()
print(x)  # global
```

### The nonlocal Keyword

To modify a closure variable (not just read it), use `nonlocal`:

```python
def make_counter():
    count = 0
    
    def counter():
        # Without nonlocal, this would create a new local 'count'
        nonlocal count
        count += 1
        return count
    
    return counter
```

Without `nonlocal`, `count += 1` would fail with `UnboundLocalError` because Python sees the assignment and assumes `count` is local.

---

## 3. The Decorator Pattern

A decorator is a function that takes a function and returns a (usually modified) function. The `@decorator` syntax is just syntactic sugar.

### Decorators Without the @ Syntax

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        result = func()
        print("After function call")
        return result
    return wrapper

def say_hello():
    print("Hello!")

# Manually apply decorator
say_hello = my_decorator(say_hello)

say_hello()
```

Output:
```
Before function call
Hello!
After function call
```

### The @ Syntax

The `@decorator` syntax does exactly the same thing:

```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        result = func()
        print("After function call")
        return result
    return wrapper

@my_decorator  # Equivalent to: say_hello = my_decorator(say_hello)
def say_hello():
    print("Hello!")

say_hello()
```

This is just syntactic sugar. The `@` syntax applies the decorator immediately after the function is defined.

### Handling Function Arguments

The wrapper needs to accept and forward arguments:

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

add(2, 3)
```

Output:
```
Calling add with args=(2, 3), kwargs={}
Result: 5
```

Using `*args` and `**kwargs` makes the wrapper work with any function signature.

---

## 4. Preserving Function Metadata with functools.wraps

There is a problem with our decorator:

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Return a greeting."""
    return f"Hello, {name}"

print(greet.__name__)  # wrapper (not greet!)
print(greet.__doc__)   # None (not "Return a greeting.")
```

The decorated function loses its identity. This breaks introspection, help text, and debugging.

### The Fix: @wraps

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Copy metadata from func to wrapper
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Return a greeting."""
    return f"Hello, {name}"

print(greet.__name__)  # greet (correct!)
print(greet.__doc__)   # Return a greeting. (correct!)
```

`@wraps(func)` copies `__name__`, `__doc__`, `__module__`, and other attributes from the original function to the wrapper.

### Always Use @wraps

This should be automatic in every decorator:

```python
from functools import wraps

def decorator_template(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Pre-processing
        result = func(*args, **kwargs)
        # Post-processing
        return result
    return wrapper
```

---

## 5. Decorators with Arguments

Sometimes we want to configure our decorator:

```python
@retry(max_attempts=3)
def fetch_data():
    pass
```

This requires an extra level of nesting.

### The Pattern

```python
from functools import wraps

def repeat(times):
    """Decorator that repeats a function call."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("Hello!")

say_hello()
```

Output:
```
Hello!
Hello!
Hello!
```

### Understanding the Nesting

```
@repeat(times=3)
def say_hello(): ...
```

Is equivalent to:

```python
def say_hello(): ...
say_hello = repeat(times=3)(say_hello)
```

1. `repeat(times=3)` is called, returning `decorator`
2. `decorator(say_hello)` is called, returning `wrapper`
3. `say_hello` now points to `wrapper`

Three levels:
- `repeat(times)`: Takes decorator arguments, returns the actual decorator
- `decorator(func)`: Takes the function, returns the wrapper
- `wrapper(*args, **kwargs)`: The actual wrapper that runs

### Making Arguments Optional

Sometimes we want a decorator to work with or without arguments:

```python
from functools import wraps

def retry(func=None, *, max_attempts=3, delay=1):
    """Retry decorator that works with or without arguments."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_error
        return wrapper
    
    if func is not None:
        # Decorator was used without arguments: @retry
        return decorator(func)
    else:
        # Decorator was used with arguments: @retry(max_attempts=5)
        return decorator

# Both work:
@retry
def fetch1():
    pass

@retry(max_attempts=5, delay=2)
def fetch2():
    pass
```

---

## 6. Class-Based Decorators

Sometimes a class is cleaner than nested functions, especially when we need to maintain state.

### Basic Class Decorator

```python
from functools import wraps

class CountCalls:
    """Count how many times a function is called."""
    
    def __init__(self, func):
        wraps(func)(self)  # Copy metadata
        self.func = func
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, **kwargs)

@CountCalls
def greet(name):
    return f"Hello, {name}"

greet("Alice")
greet("Bob")
print(greet.call_count)  # 2
```

The class implements `__call__` so instances are callable. When the decorated function is called, `__call__` is invoked.

### Class Decorator with Arguments

```python
from functools import wraps

class Retry:
    def __init__(self, max_attempts=3, exceptions=(Exception,)):
        self.max_attempts = max_attempts
        self.exceptions = exceptions
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_error = e
            raise last_error
        return wrapper

@Retry(max_attempts=5, exceptions=(ConnectionError, TimeoutError))
def fetch_data():
    pass
```

When `@Retry(...)` is used, `__init__` receives the arguments, then `__call__` receives the function.

---

## 7. Decorating Classes

Decorators can also be applied to classes.

### Class Decorator Basics

```python
def add_repr(cls):
    """Add a __repr__ method to a class."""
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"
    
    cls.__repr__ = __repr__
    return cls

@add_repr
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 30)
print(p)  # Person(name='Alice', age=30)
```

### Practical Class Decorators

```python
def singleton(cls):
    """Ensure only one instance of a class exists."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Database:
    def __init__(self, url):
        print(f"Connecting to {url}")
        self.url = url

db1 = Database("postgres://...")  # Connecting to postgres://...
db2 = Database("postgres://...")  # No print - same instance returned
print(db1 is db2)  # True
```

---

## 8. Stacking Decorators

Multiple decorators can be applied to a single function. They apply from bottom to top.

### Execution Order

```python
def decorator_a(func):
    print("Applying A")
    def wrapper(*args, **kwargs):
        print("Before A")
        result = func(*args, **kwargs)
        print("After A")
        return result
    return wrapper

def decorator_b(func):
    print("Applying B")
    def wrapper(*args, **kwargs):
        print("Before B")
        result = func(*args, **kwargs)
        print("After B")
        return result
    return wrapper

@decorator_a
@decorator_b
def greet():
    print("Hello!")

# At decoration time:
# Output: Applying B, then Applying A (bottom to top)

greet()
# Output:
# Before A
# Before B
# Hello!
# After B
# After A
```

The decorators wrap like layers of an onion:
1. `decorator_b` wraps `greet`
2. `decorator_a` wraps the result of step 1

When called, the outermost wrapper (A) runs first, calls the inner wrapper (B), which calls the original function.

---

## 9. Practical Decorator Patterns

### Pattern 1: Retry with Exponential Backoff

```python
import time
import random
from functools import wraps

def retry(
    max_attempts=3,
    initial_delay=1,
    backoff_factor=2,
    exceptions=(Exception,)
):
    """Retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # Add jitter to prevent thundering herd
                        sleep_time = delay + random.uniform(0, delay * 0.1)
                        time.sleep(sleep_time)
                        delay *= backoff_factor
            
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, initial_delay=1, exceptions=(ConnectionError,))
def fetch_api():
    # This will retry on ConnectionError
    response = requests.get("https://api.example.com")
    return response.json()
```

### Pattern 2: Caching/Memoization

```python
from functools import wraps

def memoize(func):
    """Cache function results based on arguments."""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a hashable key from arguments
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    # Expose cache for inspection/clearing
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Fast due to memoization
print(fibonacci.cache)  # See what's cached
fibonacci.clear_cache()  # Clear if needed
```

Note: Python's `functools.lru_cache` is usually better for this.

### Pattern 3: Timing

```python
import time
from functools import wraps

def timer(func):
    """Log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

slow_function()  # slow_function took 1.0012s
```

### Pattern 4: Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_second):
    """Limit how often a function can be called."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # Use list to allow nonlocal modification
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=2)
def call_api():
    print(f"API called at {time.time():.2f}")

for _ in range(5):
    call_api()  # Will space calls 0.5s apart
```

### Pattern 5: Type Checking

```python
from functools import wraps

def validate_types(**expected_types):
    """Validate argument types at runtime."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check kwargs
            for name, expected_type in expected_types.items():
                if name in kwargs:
                    if not isinstance(kwargs[name], expected_type):
                        raise TypeError(
                            f"{name} must be {expected_type.__name__}, "
                            f"got {type(kwargs[name]).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

create_user(name="Alice", age=30)  # Works
create_user(name="Alice", age="30")  # Raises TypeError
```

### Pattern 6: Authentication

```python
from functools import wraps

def require_auth(func):
    """Ensure user is authenticated before calling function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Assume request is passed as first argument
        request = args[0] if args else kwargs.get('request')
        
        if not request or not hasattr(request, 'user'):
            raise PermissionError("Authentication required")
        
        if not request.user.is_authenticated:
            raise PermissionError("User not authenticated")
        
        return func(*args, **kwargs)
    return wrapper

def require_role(role):
    """Ensure user has specific role."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            
            if not hasattr(request.user, 'role') or request.user.role != role:
                raise PermissionError(f"Role '{role}' required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_auth
@require_role("admin")
def delete_user(request, user_id):
    # Only authenticated admins can reach here
    pass
```

### Pattern 7: Logging

```python
import logging
from functools import wraps

def log_calls(logger=None, level=logging.INFO):
    """Log function calls with arguments and results."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(
                level,
                f"Calling {func.__name__}(args={args}, kwargs={kwargs})"
            )
            try:
                result = func(*args, **kwargs)
                logger.log(
                    level,
                    f"{func.__name__} returned {result!r}"
                )
                return result
            except Exception as e:
                logger.exception(f"{func.__name__} raised {e!r}")
                raise
        return wrapper
    return decorator

@log_calls()
def process_data(data):
    return len(data)
```

---

## 10. Debugging Decorated Functions

Decorated functions can be tricky to debug because the actual function is wrapped.

### Accessing the Original Function

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}"

# Access the original function
print(greet.__wrapped__)  # <function greet at 0x...>

# Call the original directly (bypass decorator)
greet.__wrapped__("Alice")
```

`@wraps` adds `__wrapped__` attribute pointing to the original function.

### Debugging Tips

1. **Print decorator execution**: Add prints to see when decorators run
2. **Check `__name__` and `__wrapped__`**: Verify the decorator is applied correctly
3. **Temporarily remove decorators**: Comment out `@decorator` lines to isolate issues
4. **Use a debugger**: Set breakpoints in the wrapper function

---

## 11. Async Decorators

For async functions, the wrapper must also be async:

```python
import asyncio
from functools import wraps

def async_timer(func):
    """Time an async function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@async_timer
async def fetch_data():
    await asyncio.sleep(1)
    return "data"

asyncio.run(fetch_data())  # fetch_data took 1.0012s
```

### Async Retry Decorator

```python
import asyncio
from functools import wraps

def async_retry(max_attempts=3, delay=1, backoff=2):
    """Retry an async function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator

@async_retry(max_attempts=3, delay=1)
async def fetch_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()
```

---

## 12. Common Mistakes

### Mistake 1: Forgetting @wraps

```python
# Bad - loses function metadata
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Good
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

### Mistake 2: Calling the Decorator Instead of Applying It

```python
# Bad - calls greet immediately!
@my_decorator()  # Note the parentheses
def greet():
    pass

# This happens when decorator doesn't take arguments
# If decorator takes no args, don't use parentheses:
@my_decorator
def greet():
    pass
```

### Mistake 3: Not Returning the Result

```python
# Bad - wrapper doesn't return anything
def decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)  # Missing return!
    return wrapper

# Good
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)  # Return the result
    return wrapper
```

### Mistake 4: Modifying Mutable Default Arguments

```python
# Bad - shared mutable state across all decorated functions
def decorator(func, cache={}):  # cache is shared!
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

# Good - each decorated function gets its own cache
def decorator(func):
    cache = {}  # Created fresh for each decoration
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper
```

---

## Summary

Decorators are functions that transform other functions. They leverage Python's treatment of functions as first-class objects and the closure mechanism to capture and extend behavior.

Key points:
- Functions are objects that can be passed around and returned
- Closures capture variables from enclosing scopes
- `@decorator` is syntactic sugar for `func = decorator(func)`
- Always use `@wraps` to preserve function metadata
- Decorators with arguments need three levels of nesting
- Stacked decorators apply from bottom to top

Practical patterns:
- Retry with exponential backoff
- Caching/memoization
- Timing and logging
- Rate limiting
- Authentication and authorization
- Input validation

For LLM applications, decorators are useful for:
- Retrying failed API calls
- Caching expensive model outputs
- Logging prompts and responses
- Rate limiting to respect API quotas
- Timing inference operations

Understanding closures and decorators also helps us understand how frameworks like FastAPI, Flask, and pytest work under the hood.
