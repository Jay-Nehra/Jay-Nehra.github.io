---
tags:
  - python
  - http
  - async
  - api
---

# httpx - Modern HTTP Client

## Part 1: Architecture

### The Mental Model: The Postal Service

Think of HTTP as a postal system:

1. **You** (client) write a **request** (letter with instructions)
2. You give it to a **transport** (postal carrier) who delivers it
3. The **server** (recipient) reads it and writes a **response** (reply letter)
4. The transport brings the response back to you

`httpx` is your interface to this postal system. It handles:
- Writing properly formatted requests
- Managing the transport (connections)
- Parsing responses

### What Problem Does This Solve?

**The raw reality**: HTTP is complex. A single request involves:

1. DNS resolution (hostname → IP address)
2. TCP connection (3-way handshake)
3. TLS handshake (for HTTPS—certificates, encryption setup)
4. Request formatting (headers, body encoding)
5. Sending bytes over the wire
6. Receiving response bytes
7. Parsing response (status, headers, body)
8. Possibly decompression (gzip)
9. Connection cleanup or pooling

You don't want to do this manually. Ever.

**Why httpx over requests?**

`requests` is the classic choice, but it's showing age:

| Feature | requests | httpx |
|---------|----------|-------|
| Sync support | ✅ | ✅ |
| Async support | ❌ | ✅ |
| HTTP/2 | ❌ | ✅ |
| Strict timeouts | Tricky | ✅ |
| Type hints | Partial | ✅ |
| Maintained | Slowly | Actively |

`httpx` is the modern replacement: same simple API, async-capable, better defaults.

### The Machinery: What Actually Happens

#### When You Call `httpx.get(url)`

This simple line does a LOT:

```python
import httpx
response = httpx.get("https://api.example.com/users")
```

**Step 1: URL Parsing**
```
https://api.example.com/users
  │       │              │
  │       │              └── Path: /users
  │       └── Host: api.example.com
  └── Scheme: https (port 443 implied)
```

**Step 2: DNS Resolution**
- Query DNS for `api.example.com`
- Get IP address (e.g., `93.184.216.34`)
- This might be cached from previous requests

**Step 3: Connection Establishment**
```
Your machine                    Server
     │                            │
     │── SYN ────────────────────>│  TCP handshake
     │<──────────────── SYN-ACK ──│
     │── ACK ────────────────────>│
     │                            │
     │── ClientHello ────────────>│  TLS handshake
     │<──────────── ServerHello ──│
     │<──────────── Certificate ──│
     │── Key Exchange ───────────>│
     │<──────────────── Finished ─│
     │                            │
     [Encrypted channel ready]
```

**Step 4: Request Sending**
```
GET /users HTTP/1.1
Host: api.example.com
User-Agent: python-httpx/0.26.0
Accept: */*
Accept-Encoding: gzip, deflate
Connection: keep-alive

[empty body for GET]
```

**Step 5: Response Receiving**
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1234
Content-Encoding: gzip

[gzipped bytes...]
```

**Step 6: Response Processing**
- Decompress gzip body
- Parse JSON (if you call `.json()`)
- Return `Response` object

**All of this happens in one line.** That's the abstraction.

#### The Client Object: Connection Pooling

When you use the module-level functions (`httpx.get()`, `httpx.post()`), each call:
1. Creates a new client
2. Makes the request
3. Closes the client

This means no connection reuse. Slow for multiple requests.

```python
# Slow: 3 new connections
httpx.get("https://api.example.com/users")
httpx.get("https://api.example.com/posts")
httpx.get("https://api.example.com/comments")
```

Using a `Client` object enables connection pooling:

```python
# Fast: 1 connection, reused 3 times
with httpx.Client() as client:
    client.get("https://api.example.com/users")
    client.get("https://api.example.com/posts")
    client.get("https://api.example.com/comments")
```

**What's happening under the hood:**

```
Request 1: [DNS] [TCP] [TLS] [Request] [Response]
Request 2:                   [Request] [Response]  ← Connection reused!
Request 3:                   [Request] [Response]  ← Connection reused!
```

The Client maintains a connection pool. When a request finishes, the connection goes back to the pool for reuse.

#### Sync vs Async: Same API, Different Engine

```python
# Synchronous (blocking)
import httpx

with httpx.Client() as client:
    response = client.get(url)  # Blocks until complete

# Asynchronous (non-blocking)
import httpx
import asyncio

async def fetch():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # Yields control while waiting

asyncio.run(fetch())
```

**What's different internally:**

- **Sync**: Uses `socket.recv()` which blocks the thread
- **Async**: Uses `asyncio` event loop, yields control during I/O wait

**When to use async:**
- Making many concurrent requests
- Building async applications (FastAPI, etc.)
- Need to do other work while waiting

**When sync is fine:**
- Simple scripts
- Few sequential requests
- CLI tools

### Key Concepts (Behavioral Definitions)

**Request**
- What we might assume: "A URL we want to access"
- What it actually means: A structured message with method, URL, headers, and optional body
- Why this matters: Headers control behavior (auth, content-type, caching)

**Response**
- What we might assume: "The data we get back"
- What it actually means: A structured message with status code, headers, and body (which may need decoding)
- Why this matters: Status codes indicate success/failure; headers contain metadata

**Status Code**
- What we might assume: "200 = good, everything else = bad"
- What it actually means: 2xx = success, 3xx = redirect, 4xx = client error, 5xx = server error
- Why this matters: Different errors need different handling (retry 503, don't retry 404)

**Timeout**
- What we might assume: "How long before giving up"
- What it actually means: There are multiple timeouts (connect, read, write, pool)
- Why this matters: Without timeouts, requests can hang forever

**Connection Pool**
- What we might assume: "Something that happens automatically"
- What it actually means: A cache of open connections, reused for subsequent requests to the same host
- Why this matters: Massive performance improvement for repeated requests

### Design Decisions: Why Is It This Way?

**Why separate `Client` from functions?**

Convenience vs performance trade-off:
- `httpx.get()` is convenient for one-off requests
- `Client()` is efficient for multiple requests

You choose based on your use case.

**Why doesn't `get()` raise on 4xx/5xx?**

```python
response = httpx.get("https://example.com/notfound")
print(response.status_code)  # 404 — no exception!
```

Design philosophy: A 404 is a valid HTTP response. The server responded successfully—it just said "not found." Whether that's an error is application-specific.

If you want exceptions:
```python
response = httpx.get(url)
response.raise_for_status()  # Raises HTTPStatusError on 4xx/5xx
```

**Why are timeouts explicit?**

`requests` had confusing timeout behavior. `httpx` is explicit:
```python
# This request will never timeout (dangerous!)
httpx.get(url)

# This has a 10-second timeout
httpx.get(url, timeout=10.0)

# Fine-grained control
httpx.get(url, timeout=httpx.Timeout(
    connect=5.0,    # Time to establish connection
    read=10.0,      # Time to read response
    write=10.0,     # Time to send request
    pool=5.0,       # Time to get connection from pool
))
```

### What Breaks If You Misunderstand

**Mistake 1: No timeout = potential hang forever**

```python
# If the server never responds, this hangs forever
response = httpx.get("https://slow-server.com/")

# Fix: Always set timeout
response = httpx.get("https://slow-server.com/", timeout=30.0)
```

**Mistake 2: Not checking status codes**

```python
response = httpx.get("https://api.com/users/999")
data = response.json()  # Might be an error response!

# Fix: Check first
response.raise_for_status()  # Raises if 4xx/5xx
data = response.json()
```

**Mistake 3: Not using connection pooling**

```python
# Slow: New connection every request
for user_id in range(100):
    response = httpx.get(f"https://api.com/users/{user_id}")

# Fast: Reuse connections
with httpx.Client() as client:
    for user_id in range(100):
        response = client.get(f"https://api.com/users/{user_id}")
```

**Mistake 4: Blocking async code**

```python
async def fetch_all(urls):
    results = []
    for url in urls:
        response = await client.get(url)  # Sequential! Not parallel!
        results.append(response)
    return results

# Fix: Use asyncio.gather for parallelism
async def fetch_all(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return await asyncio.gather(*tasks)
```

---

## Part 2: Scenarios

### Scenario 1: Basic GET Request

The simplest case:

```python
import httpx

# One-off request
response = httpx.get(
    "https://api.github.com/users/octocat",
    timeout=10.0  # Always set timeout!
)

# Check success
if response.status_code == 200:
    user = response.json()
    print(user["login"])
else:
    print(f"Error: {response.status_code}")

# Or use raise_for_status for exceptions
try:
    response.raise_for_status()
    user = response.json()
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Request failed: {e}")
```

**What `.json()` does:**
1. Reads response body bytes
2. Decodes using charset from Content-Type header (usually UTF-8)
3. Parses JSON into Python dict/list
4. Raises `json.JSONDecodeError` if invalid JSON

### Scenario 2: POST with JSON Body

Sending data to an API:

```python
import httpx

# POST JSON data
response = httpx.post(
    "https://api.example.com/users",
    json={"name": "Jay", "email": "jay@example.com"},  # Automatically serialized
    timeout=10.0
)

# Using json= parameter:
# - Sets Content-Type: application/json
# - Calls json.dumps() on your data
# - Encodes to UTF-8 bytes

# Alternative: Manual body
response = httpx.post(
    "https://api.example.com/users",
    content=b'{"name": "Jay"}',  # Raw bytes
    headers={"Content-Type": "application/json"}
)
```

**Other body types:**

```python
# Form data (like HTML forms)
response = httpx.post(
    url,
    data={"username": "jay", "password": "secret"}
    # Content-Type: application/x-www-form-urlencoded
)

# File upload
with open("photo.jpg", "rb") as f:
    response = httpx.post(
        url,
        files={"image": f}
        # Content-Type: multipart/form-data
    )
```

### Scenario 3: Using a Client (The Right Way)

For real applications, always use a Client:

```python
import httpx
from contextlib import contextmanager

# Configuration
BASE_URL = "https://api.example.com"
API_TOKEN = "your-token-here"

@contextmanager
def get_api_client():
    """Create configured API client."""
    client = httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=30.0,
    )
    try:
        yield client
    finally:
        client.close()

# Usage
with get_api_client() as client:
    # All requests use base_url, headers, timeout
    users = client.get("/users").json()
    posts = client.get("/posts").json()
    
    # Override per-request
    response = client.get("/slow-endpoint", timeout=60.0)
```

**Client configuration options:**

```python
client = httpx.Client(
    # Base URL prepended to relative paths
    base_url="https://api.example.com",
    
    # Default headers for all requests
    headers={
        "Authorization": "Bearer token",
        "User-Agent": "MyApp/1.0",
    },
    
    # Default timeout
    timeout=30.0,
    
    # Cookie handling
    cookies={"session": "abc123"},
    
    # Follow redirects (default True for httpx)
    follow_redirects=True,
    
    # HTTP/2 support
    http2=True,
    
    # Proxy configuration
    proxy="http://proxy.example.com:8080",
    
    # SSL certificate verification
    verify=True,  # Set to False to skip (dangerous!)
)
```

### Scenario 4: Async Client for Concurrent Requests

When you need to make many requests efficiently:

```python
import httpx
import asyncio

async def fetch_user(client: httpx.AsyncClient, user_id: int) -> dict:
    """Fetch a single user."""
    response = await client.get(f"/users/{user_id}")
    response.raise_for_status()
    return response.json()

async def fetch_all_users(user_ids: list[int]) -> list[dict]:
    """Fetch multiple users concurrently."""
    async with httpx.AsyncClient(
        base_url="https://api.example.com",
        timeout=30.0
    ) as client:
        # Create all tasks
        tasks = [fetch_user(client, uid) for uid in user_ids]
        
        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        users = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Failed: {result}")
            else:
                users.append(result)
        
        return users

# Run it
users = asyncio.run(fetch_all_users([1, 2, 3, 4, 5]))
```

**Why this is fast:**

```
Sequential (sync):    [Req1]──────[Req2]──────[Req3]────── Total: 3x
Concurrent (async):   [Req1]
                      [Req2]       All overlapping!
                      [Req3]                               Total: ~1x
```

### Scenario 5: Error Handling and Retries

Robust request handling:

```python
import httpx
import time
from typing import TypeVar, Callable

T = TypeVar("T")

def retry_request(
    func: Callable[[], T],
    max_attempts: int = 3,
    backoff: float = 1.0,
    retryable_status: tuple = (500, 502, 503, 504),
) -> T:
    """Retry a request with exponential backoff."""
    
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            response = func()
            
            # Check if we should retry based on status
            if hasattr(response, 'status_code'):
                if response.status_code in retryable_status:
                    raise httpx.HTTPStatusError(
                        f"Retryable status: {response.status_code}",
                        request=response.request,
                        response=response
                    )
            
            return response
            
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                sleep_time = backoff * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
    
    raise last_exception

# Usage
def make_request():
    with httpx.Client(timeout=10.0) as client:
        response = client.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()

data = retry_request(make_request)
```

**Exception hierarchy:**

```
httpx.HTTPError (base)
├── httpx.RequestError (network failures)
│   ├── httpx.ConnectError (can't connect)
│   ├── httpx.TimeoutException (timeout)
│   ├── httpx.ReadError (error reading response)
│   └── httpx.WriteError (error sending request)
│
└── httpx.HTTPStatusError (4xx/5xx responses)
    ├── response.status_code
    └── response.text
```

### Production Patterns

#### Pattern 1: API Client Class

```python
import httpx
from dataclasses import dataclass
from typing import Any

@dataclass
class APIError(Exception):
    status_code: int
    message: str
    response: httpx.Response

class APIClient:
    """Typed API client with error handling."""
    
    def __init__(self, base_url: str, api_key: str):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
    
    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make request and handle response."""
        response = self._client.request(method, path, **kwargs)
        
        if response.status_code >= 400:
            raise APIError(
                status_code=response.status_code,
                message=response.text,
                response=response,
            )
        
        return response.json()
    
    def get_user(self, user_id: int) -> dict:
        return self._request("GET", f"/users/{user_id}")
    
    def create_user(self, name: str, email: str) -> dict:
        return self._request("POST", "/users", json={"name": name, "email": email})
    
    def close(self):
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

# Usage
with APIClient("https://api.example.com", "my-key") as api:
    user = api.get_user(123)
    new_user = api.create_user("Jay", "jay@example.com")
```

#### Pattern 2: Rate limiting

```python
import httpx
import time
from threading import Lock

class RateLimitedClient:
    """Client that respects rate limits."""
    
    def __init__(self, base_url: str, requests_per_second: float = 10):
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self._min_interval = 1.0 / requests_per_second
        self._last_request = 0.0
        self._lock = Lock()
    
    def _wait_for_rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.time()
    
    def get(self, path: str, **kwargs) -> httpx.Response:
        self._wait_for_rate_limit()
        return self._client.get(path, **kwargs)
    
    def close(self):
        self._client.close()
```

#### Pattern 3: Response caching

```python
import httpx
import hashlib
from functools import lru_cache
from typing import Optional

class CachingClient:
    """Client with simple in-memory caching."""
    
    def __init__(self, base_url: str, cache_size: int = 100):
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self._cache: dict[str, tuple[int, str]] = {}
        self._cache_size = cache_size
    
    def _cache_key(self, method: str, url: str) -> str:
        return hashlib.md5(f"{method}:{url}".encode()).hexdigest()
    
    def get(self, path: str, use_cache: bool = True) -> httpx.Response:
        key = self._cache_key("GET", path)
        
        if use_cache and key in self._cache:
            status, text = self._cache[key]
            # Return cached response (simplified)
            response = self._client.build_request("GET", path)
            # Note: This is simplified; real caching is more complex
        
        response = self._client.get(path)
        
        if response.status_code == 200:
            self._cache[key] = (response.status_code, response.text)
            # Evict old entries if cache too large
            if len(self._cache) > self._cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        
        return response
```

### What Breaks: Common Mistakes

**1. Forgetting to close the client**

```python
# Memory leak: connections never closed
client = httpx.Client()
client.get(url)
# ... program continues, client never closed

# Fix: Use context manager
with httpx.Client() as client:
    client.get(url)
# Client automatically closed
```

**2. Using sync client in async code**

```python
async def handler():
    response = httpx.get(url)  # BLOCKS the event loop!

# Fix: Use AsyncClient
async def handler():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # Non-blocking
```

**3. Ignoring SSL errors unsafely**

```python
# DANGEROUS: Disables all SSL verification
client = httpx.Client(verify=False)

# If you must (e.g., testing with self-signed certs):
import certifi
client = httpx.Client(verify="/path/to/custom/ca-bundle.crt")
```

**4. Not handling timeouts**

```python
try:
    response = httpx.get(url, timeout=5.0)
except httpx.TimeoutException:
    # Connection or read took too long
    print("Request timed out")
except httpx.ConnectError:
    # Couldn't establish connection
    print("Could not connect to server")
```

---

## Summary: The Mental Checklist

1. **One request or many?**
   - One: `httpx.get()` is fine
   - Many: Use `Client()` for connection pooling

2. **Sync or async?**
   - Scripts/CLI: Sync is simpler
   - Web apps/many concurrent: Async

3. **Did I set a timeout?**
   - Always. Default is no timeout (dangerous).

4. **Did I check the status code?**
   - Either check explicitly or use `raise_for_status()`

5. **Am I handling errors?**
   - Network errors: `RequestError`
   - HTTP errors: `HTTPStatusError`
   - Both need handling for robust code

6. **Should I retry?**
   - 5xx: Usually yes (server temporary failure)
   - 429: Yes, with backoff (rate limited)
   - 4xx: Usually no (client error won't fix itself)
