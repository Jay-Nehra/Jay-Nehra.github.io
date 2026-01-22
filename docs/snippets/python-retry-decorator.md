# Python: Retry Decorator with Exponential Backoff

**Problem:** API calls fail randomly due to network issues or rate limiting. Need automatic retries with increasing delays.

## The Snippet

```python
import time
from functools import wraps

def retry(max_attempts=3, base_delay=1, backoff_factor=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt failed, re-raise the exception
                        raise
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator

# Usage
@retry(max_attempts=3, base_delay=2)
def fetch_data(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()

# Will retry with delays: 2s, 4s, 8s
data = fetch_data("https://api.example.com/data")
```

## How It Works

1. **First attempt** fails → Wait 2 seconds
2. **Second attempt** fails → Wait 4 seconds (2 × 2)
3. **Third attempt** fails → Wait 8 seconds (4 × 2)
4. **Fourth attempt** fails → Raise the exception (max attempts reached)

**Exponential backoff** prevents hammering a struggling server and improves success rates.

## Customization Examples

### Retry Only Specific Exceptions

```python
@retry(max_attempts=3, exceptions=(requests.RequestException, TimeoutError))
def api_call():
    # Only retries network errors, not ValueError etc.
    pass
```

### Longer Initial Delay

```python
@retry(max_attempts=5, base_delay=5, backoff_factor=2)
def slow_api():
    # Delays: 5s, 10s, 20s, 40s
    pass
```

### Linear Backoff Instead of Exponential

```python
@retry(max_attempts=4, base_delay=3, backoff_factor=1)
def linear_retry():
    # Delays: 3s, 3s, 3s (same delay each time)
    pass
```

## Async Version

For async functions, use this version:

```python
import asyncio
from functools import wraps

def async_retry(max_attempts=3, base_delay=1, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"Retry {attempt + 1}/{max_attempts} in {delay}s")
                    await asyncio.sleep(delay)
        
        return wrapper
    return decorator

# Usage
@async_retry(max_attempts=3, base_delay=2)
async def fetch_async(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()
```

## When to Use

✅ **External API calls** — Network can be flaky  
✅ **Database connection retries** — Transient connection issues  
✅ **File system operations** — NFS, cloud storage intermittent failures  
✅ **Distributed system calls** — Microservices, message queues

## When NOT to Use

❌ **User-facing requests** — Don't make users wait 14+ seconds  
❌ **Write operations** — Risk of duplicate writes (payments, orders)  
❌ **Operations with side effects** — Retrying might cause unintended actions  
❌ **Already idempotent retry logic** — Don't double-retry

## Improvements to Consider

### Add Jitter

Prevent thundering herd problem:

```python
import random

delay = base_delay * (backoff_factor ** attempt)
jittered_delay = delay * (0.5 + random.random())  # ±50% randomness
time.sleep(jittered_delay)
```

### Add Logging

```python
import logging

logger = logging.getLogger(__name__)

@retry(max_attempts=3)
def func():
    logger.info(f"Attempt {attempt + 1} failed: {e}")
```

### Return Metadata

```python
def wrapper(*args, **kwargs):
    for attempt in range(max_attempts):
        try:
            result = func(*args, **kwargs)
            return {'data': result, 'attempts': attempt + 1}
        except:
            # retry logic
```

## Alternative Libraries

If you need more features, consider:

- **[tenacity](https://github.com/jd/tenacity)** — Full-featured retry library
- **[backoff](https://github.com/litl/backoff)** — Another popular option

But for simple cases, this snippet is enough.

## Related

- [Understanding Python Async/Await](../notes/python/async-explained.md)
- [Handling API Rate Limits](../notes/api-rate-limiting.md)

---

*Last updated: January 15, 2026*