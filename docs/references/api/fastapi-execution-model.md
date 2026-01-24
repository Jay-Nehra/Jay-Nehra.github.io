# FastAPI Execution Model for LLM Applications

This document explains how FastAPI actually works at runtime — not the syntax, but the execution mechanics. If you understand what happens when a request arrives, where your code runs, and what resources are shared, you will stop making the class of bugs that only appear in production.
---

## 1. What Is a Server, Really

Before we talk about FastAPI, we need to establish what a server actually is, because most confusion stems from treating servers like scripts.

A script is a program that starts, runs top to bottom, and exits. When you run a Python script, the interpreter creates a process, executes your code sequentially, and then terminates. Memory is reclaimed. State disappears. Every run begins from a clean slate.

A server is fundamentally different. A server is a program that starts and then waits. It does not exit. It stays alive indefinitely, reacting to external events over time. When you start an API server, the Python interpreter creates a process that allocates memory, opens network sockets, initializes data structures, and then enters a loop that never ends. That loop waits for requests, handles them, and then waits again. Hours later, it is still the same process, with the same memory, the same global variables, the same objects, and the same accumulated history.

This single difference — the process does not exit — is the root of almost every conceptual pitfall in API development.

### The Runtime Concept

When people talk about "runtime" in the context of servers, they mean the living execution environment of your program over time. This is not a file or a package. It is a state. For Python, the runtime includes the interpreter process itself, the memory it has allocated, the objects currently alive in that memory, the threads or event loop that are running, the open file descriptors and sockets, and the scheduling rules that decide what runs next.

In a short script, the runtime is born, does its work, and dies in seconds. You never feel it.

In an API server, the runtime lives for hours or days. Everything you create in that process persists. Mistakes accumulate. Bugs compound. A memory leak that loses 1MB per request becomes a crash after 10,000 requests. A global variable that gets overwritten becomes a security breach when two users' requests overlap.

This is why tutorials that teach you to "just call the function" miss the point entirely. In a server, you are not calling functions. You are registering handlers that the runtime will invoke later, possibly thousands of times, possibly concurrently, under conditions you do not directly control.

### Why "Works Locally" Means Nothing

When you test an API locally, you typically send one request at a time. The runtime starts fresh each time you restart the server. Memory is abundant. Network latency is zero. There is no contention for resources.

In production, everything changes. Hundreds of requests arrive simultaneously. The runtime has been alive for days. Memory pressure is real. Downstream services are slow. Rate limits exist. The problems that matter only appear under these conditions.

This is why understanding the execution model is not optional. The execution model is the difference between "it works on my machine" and "it survives production."

---

## 2. What FastAPI Actually Is

When you write `app = FastAPI()`, you are not starting a server. You are not opening sockets. You are not handling requests. You are creating a configuration object that describes how requests should be handled if they arrive.

That object contains mappings from paths and HTTP methods to Python functions, rules for validation, dependency injection logic, and lifecycle hooks. It is inert by itself. Nothing runs because of it.

### FastAPI as a Configuration Object

Think of FastAPI as a recipe book, not a kitchen. The recipe book describes what dishes exist and how to prepare them. But the recipe book does not cook anything. You need a kitchen (the ASGI server) to actually prepare food (handle requests).

When you decorate a function with `@app.get("/ping")`, you are adding an entry to this recipe book:

```python
app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok"}
```

At this point, the `ping` function exists and is registered with the app. But no server is running. No requests can be handled. The function will never be called until something else starts the actual server.

### Uvicorn's Role

The thing that actually runs the server is the ASGI server, typically Uvicorn. ASGI stands for Asynchronous Server Gateway Interface — it is a specification for how Python web applications communicate with web servers.

When you run `uvicorn main:app`, Uvicorn does the following:

1. Starts a Python process
2. Creates an event loop using asyncio
3. Opens a listening socket on the specified host and port
4. Creates a thread pool for running blocking code
5. Imports your module and evaluates `app = FastAPI()`
6. Begins accepting connections on the socket

At this point, a runtime exists. The process is alive. The event loop is spinning. Requests can now be handled.

### The Handoff: Parsing, Routing, Execution

When an HTTP request arrives over the network, several layers process it before your code runs.

First, Uvicorn reads bytes from the socket and parses them into an HTTP request. This parsing happens entirely within Uvicorn's internals. You do not control it, and you do not need to understand the byte-level details. What matters is that after this step, the request is no longer raw bytes — it is a structured object with a method, path, headers, and body.

Next, Uvicorn hands this structured request to your FastAPI application. This is the handoff point. Below this line is networking and protocol handling. Above this line is application logic.

FastAPI (via Starlette, which it is built on) now does routing. It matches the request method and path against registered routes. If you requested `GET /ping`, FastAPI finds the `ping` function you registered.

Before executing the handler, FastAPI performs validation and dependency resolution. If your handler declares parameters that come from the path, query string, headers, or body, FastAPI uses Pydantic to validate and parse those inputs into Python objects. If validation fails, your handler is never run — the client receives an error response.

Only after all of this does FastAPI decide how to actually execute your handler. This decision is where concurrency behavior is determined.

### Starlette Underneath

FastAPI is built on top of Starlette, which provides the low-level ASGI handling, routing, middleware, and request/response objects. FastAPI adds dependency injection, automatic validation, serialization, and OpenAPI documentation generation.

You rarely need to think about Starlette directly, but understanding that it exists helps explain why FastAPI feels "declarative." You are not telling Python to "run this function now." You are declaring intent: "when a request like this arrives, here is what should happen."

---

## 3. Request Execution: The Complete Trace

Let us trace exactly what happens when a request arrives, step by step. This is the execution model you need to internalize.

### Bytes Arrive

A client sends an HTTP request to your server. At the lowest level, this is bytes arriving on a TCP socket. Uvicorn's event loop is notified that data is available. Uvicorn reads the bytes and parses them according to the HTTP protocol.

The result is a structured request: method `GET`, path `/ping`, headers like `Content-Type: application/json`, and possibly a body.

### Routing

Uvicorn passes this structured request to your FastAPI app. FastAPI looks up which handler matches the method and path. For `GET /ping`, it finds your `ping` function.

### Validation and Dependency Resolution

Before running the handler, FastAPI inspects its signature. If you declared parameters, FastAPI extracts values from the request and validates them.

For example, if your handler is:

```python
@app.get("/users/{user_id}")
def get_user(user_id: int, include_profile: bool = False):
    ...
```

FastAPI will:
- Extract `user_id` from the path and validate it is an integer
- Extract `include_profile` from query parameters and validate it is a boolean
- If validation fails, return a 422 error without running your handler

FastAPI also resolves dependencies at this stage. If your handler uses `Depends(...)`, those dependency functions are called first, and their results are passed to your handler.

All of this happens before your handler code runs. This is important: the request has been fully parsed and validated before you see it.

### The Sync/Async Branching Decision

Now comes the critical moment. FastAPI has a handler it wants to execute. It looks at the handler and asks one question:

Is this a coroutine function (`async def`) or a regular function (`def`)?

This is not about syntax preference. It is about scheduling policy. The answer determines where and how your code runs.

If the handler is defined with `def`, FastAPI assumes it may block. To protect the event loop, it schedules the function to run in a worker thread.

If the handler is defined with `async def`, FastAPI assumes it will cooperate with the event loop. It schedules the handler as a task on the event loop itself.

### Pseudo-Code Trace: Sync Handler

Let us trace what happens with a sync handler:

```python
@app.get("/ping")
def ping():
    return {"status": "ok"}
```

Conceptually, FastAPI does something like this:

```python
handler = ping  # regular function

# Cannot run directly on event loop — would block
# Submit to thread pool
future = event_loop.run_in_executor(
    thread_pool_executor,
    handler
)

# Event loop continues, waits for future to complete
result = await future

# Serialize result and return response
return JSONResponse(result)
```

The key points:
- The handler runs in a worker thread, not on the event loop
- The event loop remains free to handle other requests
- When the thread finishes, the result is returned to the event loop

### Pseudo-Code Trace: Async Handler

Now compare with an async handler:

```python
@app.get("/ping")
async def ping():
    return {"status": "ok"}
```

Conceptually, FastAPI does this:

```python
handler = ping  # coroutine function

# Call the function — this returns a coroutine, not a result
coroutine = handler()

# Schedule the coroutine as a task on the event loop
task = asyncio.create_task(coroutine)

# Wait for the task to complete
result = await task

# Serialize result and return response
return JSONResponse(result)
```

The key points:
- The handler runs on the event loop itself
- No worker thread is involved
- The handler must cooperate by yielding at `await` points

---

## 4. Sync Handlers: Thread Pool Execution

When you write a sync handler (`def`), FastAPI runs it in a thread pool. Understanding how this works is essential for predicting behavior under load.

### What run_in_executor Actually Does

The `run_in_executor` mechanism is a standard asyncio feature. It allows synchronous, potentially blocking code to run in a separate thread while the event loop continues.

When FastAPI calls `run_in_executor`:
1. The handler function is placed in a queue
2. One worker thread from the pool picks it up
3. That thread executes the function fully, blocking until it returns
4. The result is placed in a Future object
5. The event loop is notified that the Future is ready

While the worker thread is executing:
- The event loop is not blocked
- Other requests can be accepted and routed
- Async handlers can run
- Other sync handlers can run on other threads (if available)

The event loop acts as a coordinator, not an executor.

### Thread Pool Size, Configuration, and Limits

The thread pool has a finite number of threads. By default, this is often set to something like `min(32, cpu_count + 4)` in Python's concurrent.futures, though ASGI servers may configure this differently.

You can configure the thread pool size, but increasing it is not a free win. More threads mean:
- More memory usage (each thread has its own stack)
- More context switching overhead
- More contention for shared resources

At some point, adding threads makes performance worse, not better.

The important insight is that concurrency for sync handlers is bounded by the thread pool size. If you have 10 threads and 100 concurrent requests, 90 requests are waiting.

### Thread Pool Exhaustion: Symptoms, Causes, Solutions

Thread pool exhaustion happens when all worker threads are busy and new requests must wait for a thread to become available.

**Symptoms:**
- Latency increases linearly with load
- CPU usage may be moderate (threads are waiting, not computing)
- Memory is stable
- Logs show requests completing, just slowly

**Causes:**
- Too many concurrent requests for the pool size
- Handlers that block for too long (slow database queries, slow HTTP calls)
- Handlers that do CPU-intensive work

**Solutions:**
- Increase thread pool size (limited benefit)
- Make handlers faster (reduce blocking time)
- Convert to async handlers with async I/O
- Add backpressure (reject requests when overloaded)

**Recognition pattern:** If latency grows steadily under load but nothing crashes, suspect thread pool exhaustion.

### When Sync Is Correct

Sync handlers are not wrong. They are appropriate when:
- Your code uses blocking libraries that have no async equivalent
- The blocking time is short and predictable
- You have sized your thread pool appropriately
- You prefer simplicity over maximum throughput

Many production systems run entirely on sync handlers and work fine. The key is understanding the tradeoffs.

---

## 5. Async Handlers: Event Loop Execution

When you write an async handler (`async def`), FastAPI runs it directly on the event loop. This enables high concurrency for I/O-bound work, but introduces different failure modes.

### Coroutines as Paused Computations

An `async def` function does not run when you call it. Instead, calling it returns a coroutine object — essentially a paused computation that knows how to resume itself.

```python
async def ping():
    return {"status": "ok"}

coro = ping()  # Does NOT run the body
# coro is a coroutine object, not {"status": "ok"}
```

To actually run the coroutine, you must either `await` it or schedule it as a task. The event loop is responsible for driving coroutines forward.

### create_task Mechanics

When FastAPI schedules an async handler, it creates a task:

```python
task = asyncio.create_task(coroutine)
```

A Task wraps a coroutine and registers it with the event loop. The event loop will:
1. Start executing the coroutine
2. Run until it hits an `await`
3. Pause the task and switch to other work
4. Resume the task when the awaited operation completes
5. Repeat until the coroutine finishes

This is cooperative multitasking. Tasks voluntarily yield control at `await` points.

### What await Actually Does

The `await` keyword is not decoration. It is the mechanism by which tasks yield control.

When you write:

```python
async def fetch_data():
    response = await http_client.get("https://api.example.com/data")
    return response.json()
```

The `await` does two things:
1. Tells the event loop: "I am waiting for this operation. You can run other tasks."
2. When the operation completes, tells the event loop: "Resume me here."

Without `await`, there is no yielding. The task runs continuously until it returns. If it blocks (via non-async I/O, CPU work, or `time.sleep`), the event loop cannot switch tasks.

### Event Loop Starvation: Symptoms, Causes, Solutions

Event loop starvation happens when a task blocks the event loop, preventing other tasks from running.

**Symptoms:**
- Sudden, system-wide latency spikes
- All requests slow down at once (not gradual)
- CPU may look idle (the loop is blocked waiting)
- One slow request affects all concurrent requests

**Causes:**
- Blocking I/O inside async handlers (sync HTTP clients, `time.sleep`)
- CPU-intensive work without yielding
- Long loops without `await` points
- Calling sync libraries that block

**Solutions:**
- Use async libraries for all I/O
- Offload blocking work to thread pool with `run_in_executor`
- Offload CPU work to process pool
- Add periodic `await asyncio.sleep(0)` in long loops

**Recognition pattern:** If latency spikes affect all requests simultaneously and the system feels "stuck," suspect event loop starvation.

### Why Event Loop Starvation Is Worse Than Thread Pool Exhaustion

Thread pool exhaustion degrades gracefully. Requests queue up and complete slowly, but the system remains responsive. You can still accept new connections. Health checks pass.

Event loop starvation is catastrophic. When the loop is blocked, nothing runs. New connections are not accepted. Existing responses are not sent. Health checks fail. Load balancers mark your server as dead.

This is why blocking code in async handlers is such a serious bug. It looks like working code but fails suddenly under load.

---

## 6. Blocking Operations in Async Code

The most common mistake in FastAPI applications is putting blocking operations inside async handlers. This section explains why this happens, how to recognize it, and how to fix it.

### The time.sleep Example

Consider this handler:

```python
import time

@app.get("/slow")
async def slow():
    time.sleep(1)  # Simulating slow work
    return {"status": "done"}
```

This looks harmless. In testing, it works. With one request, it takes one second. Everything seems fine.

Now send two requests simultaneously.

Request A arrives. The event loop schedules the `slow` task and starts executing it. The task calls `time.sleep(1)`. This is a blocking system call. It does not know about async. It does not yield. It blocks the current thread — which is the event loop thread — for one second.

Request B arrives during that second. Uvicorn can accept the connection at the OS level, but the event loop cannot process it. Routing, scheduling, response handling — all paused.

After one second, Request A's sleep finishes. The task completes. The event loop wakes up and processes Request B. But now Request B also calls `time.sleep(1)` and blocks the loop again.

With 100 concurrent requests, they serialize. One per second. Throughput collapses. Latency explodes. Nothing crashes. This is the worst kind of failure.

### The Sync HTTP Client Example

The same problem occurs with sync HTTP clients:

```python
import requests

@app.get("/generate")
async def generate(prompt: str):
    response = requests.post(
        "https://api.llm-provider.com/generate",
        json={"prompt": prompt}
    )
    return response.json()
```

This is extremely common in LLM applications. The `requests` library is blocking. When `requests.post` runs, it blocks until the HTTP response arrives. If the LLM takes 5 seconds to respond, the event loop is frozen for 5 seconds.

The danger is psychological. Making an HTTP request feels like "waiting on I/O," which sounds like something async should handle. But async only works if you yield control. The `requests` library does not yield.

### Why This Passes Tests and Fails Production

In testing:
- You send one request at a time
- Each request works correctly
- Response times are as expected
- No errors appear

In production:
- Many requests arrive simultaneously
- They serialize instead of parallelize
- Latency grows with concurrency
- Throughput is capped at one request per blocking duration

This gap between testing and production is why execution model understanding matters.

### Correct Solution 1: Async HTTP Clients

The best solution is to use an async HTTP client:

```python
import httpx

@app.get("/generate")
async def generate(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.llm-provider.com/generate",
            json={"prompt": prompt}
        )
    return response.json()
```

When `await client.post(...)` runs, it yields control to the event loop. The event loop can process other requests while waiting for the HTTP response. When the response arrives, the task resumes.

This is cooperative waiting. The event loop stays healthy. Concurrency scales with I/O.

### Correct Solution 2: Offload to Thread Pool

If you must use a blocking library, explicitly offload it:

```python
import asyncio
import requests

@app.get("/generate")
async def generate(prompt: str):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,  # Uses default thread pool
        lambda: requests.post(
            "https://api.llm-provider.com/generate",
            json={"prompt": prompt}
        )
    )
    return response.json()
```

This runs the blocking call in a worker thread, just like a sync handler would. The event loop remains responsive.

This is less efficient than true async I/O (you're using threads), but it's correct. It prevents event loop starvation.

### Mental Image: The Intersection

Think of the event loop as a traffic controller at an intersection.

Async I/O calls are like cars that pull into a waiting lane and signal when they're ready to proceed. The controller can manage many of them at once.

Blocking calls are like a truck that parks in the middle of the intersection and waits for a delivery. No one else moves until it leaves.

It doesn't matter that the truck is "waiting." It's still blocking the intersection.

---

## 7. Lifespan and Shared Resources

Now that you understand request execution, we can address a higher-level question: where should long-lived things live?

### Process Scope vs Request Scope

Because your API server is a long-running process, anything you create has a lifetime. The two important lifetimes are:

**Process scope:** Objects that exist for the entire life of the server. Created at startup, destroyed at shutdown. Shared by all requests.

**Request scope:** Objects that exist for one request. Created when the request arrives, destroyed when it completes. Isolated per request.

The fundamental rule is: put things in the correct scope.

### What Belongs in Lifespan

Lifespan is FastAPI's mechanism for managing process-scoped resources. Things that belong here:

**ML models:** Loading a model takes seconds and hundreds of megabytes. You cannot reload per request. Load once at startup.

**HTTP clients:** Creating an HTTP client involves connection pools, TLS context, and buffers. Reusing clients across requests gives connection reuse and stable performance.

**Database connection pools:** Opening database connections is expensive. Pools maintain a set of connections that are borrowed and returned per request.

**Tokenizers and artifacts:** Anything expensive to load and safe to share.

**Caches:** In-memory caches must persist across requests to be useful.

Here's how lifespan looks in FastAPI:

```python
from contextlib import asynccontextmanager
import httpx

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create resources
    app.state.http_client = httpx.AsyncClient()
    app.state.model = load_ml_model()
    
    yield  # Server runs here
    
    # Shutdown: cleanup resources
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(request: Request, text: str):
    client = request.app.state.http_client
    model = request.app.state.model
    # Use shared resources
```

### What Belongs in Dependencies

Dependencies are FastAPI's mechanism for managing request-scoped resources. Things that belong here:

**API keys and auth context:** Extracted from headers, different per request, must not leak across requests.

**User identity:** Determined per request from authentication.

**Database transactions:** Opened per request, committed or rolled back when the request completes.

**Request-specific configuration:** Feature flags, rate limit state, etc.

Here's how dependencies work:

```python
from fastapi import Depends, Header

async def get_current_user(authorization: str = Header(...)):
    # Validate token, return user
    return validate_token(authorization)

@app.get("/profile")
async def profile(user: User = Depends(get_current_user)):
    return {"user": user.name}
```

Each request gets its own call to `get_current_user`. The result is not shared.

### Cross-Request Contamination Bugs

If you put request-scoped data in process scope, you get cross-request contamination. This is a security and correctness failure.

Consider this broken code:

```python
current_user = None  # Global variable

@app.get("/profile")
async def profile(authorization: str = Header(...)):
    global current_user
    current_user = validate_token(authorization)
    # ... do some async work ...
    return {"user": current_user.name}  # Bug!
```

If Request A and Request B overlap:
1. Request A sets `current_user` to User A
2. Request A does async work (yields control)
3. Request B sets `current_user` to User B
4. Request A resumes and reads `current_user` — now User B!

User A sees User B's data. This is a silent, serious bug.

### Lazy Initialization Race Conditions

Another common mistake is lazy initialization of shared resources:

```python
_client = None

def get_client():
    global _client
    if _client is None:
        _client = create_expensive_client()
    return _client
```

If two requests call `get_client` simultaneously before initialization:
1. Request A checks: `_client is None` → True
2. Request B checks: `_client is None` → True
3. Request A starts creating client
4. Request B starts creating client
5. Both create clients, one overwrites the other
6. Resources may leak, state may be inconsistent

Lifespan avoids this by initializing once, before any requests.

### Mental Image: The Factory

Imagine the server as a factory that never closes.

When the factory opens for the day, it sets up heavy machinery, power systems, and shared tools. That's lifespan.

Each customer order gets its own paperwork, measurements, and instructions. That's request scope.

Workers use the shared machines to process individual orders. They must not scribble customer-specific notes on the machines themselves.

---

## 8. Background Tasks vs Real Jobs

FastAPI provides "background tasks" that run after the response is sent. Understanding what these actually are — and are not — is critical for LLM applications.

### FastAPI Background Tasks: Still In-Process

When you add a background task, you're saying: "run this after the response."

```python
from fastapi import BackgroundTasks

@app.post("/submit")
async def submit(data: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_data, data)
    return {"status": "accepted"}

def process_data(data: str):
    # This runs after response is sent
    ...
```

The client receives the response immediately. The background task runs afterward.

But here's what people miss: **background tasks run in the same process**.

They:
- Share memory with request handlers
- Use the same event loop or thread pool
- Compete for CPU, network, and downstream resources
- Can still block the event loop if async and blocking

Sending the response does not free resources. It only ends the client's wait.

### Why "Background" Is Misleading

The term "background" implies isolation. It suggests the work is happening "somewhere else." This is false.

Background tasks are foreground work that happens to start after the response. From the server's perspective, they are just more work competing for the same resources.

If a background task:
- Blocks the event loop → all requests stall
- Uses all thread pool threads → sync handlers queue
- Hammers a downstream API → rate limits affect everyone

These effects are identical to in-request work.

### The Three Execution Zones

To reason about this correctly, think of three zones:

**Zone 1: Request path.** Latency-critical, tightly bounded. Should do the minimum to accept and respond.

**Zone 2: In-process background.** Still inside the API runtime. Okay for small, bounded tasks. Dangerous for heavy work.

**Zone 3: External jobs.** Separate process or system. Own capacity, own failure modes, own lifecycle.

FastAPI gives you Zones 1 and 2. It does not solve Zone 3.

### When to Externalize to Job Queues

Heavy, long-running, or resource-intensive work belongs outside the API process.

Signs you need external jobs:
- Work takes more than a few seconds
- Work is CPU-intensive
- Work may fail and need retry
- Work should survive server restarts
- Work should not affect API latency

For LLM applications, this often means:
- Batch inference → external job queue
- Document ingestion for RAG → external pipeline
- Model fine-tuning → separate service
- Long-running agent tasks → job with status polling

The API's responsibility is to accept the work, validate it, enqueue it, and return immediately. Actual execution happens elsewhere.

---

## 9. Batch vs Online Inference

LLM applications often have two modes: real-time inference for individual requests, and batch processing for large datasets. Understanding how these differ is essential for API design.

### Different Load Shapes, Different Constraints

**Online inference** is characterized by:
- Many independent requests
- Low latency requirements (< 1 second ideal)
- Fairness matters (one user shouldn't starve others)
- Each request is small

**Batch inference** is characterized by:
- One request with many items
- Throughput matters more than latency
- Can run for minutes or hours
- Memory pressure is high

These are fundamentally different load shapes. Treating them the same causes problems.

### Why Batch Cannot Be "A Slow Request"

A common mistake is implementing batch as a synchronous loop inside a request handler:

```python
@app.post("/batch")
async def batch_process(items: List[str]):
    results = []
    for item in items:
        result = await llm_client.generate(item)
        results.append(result)
    return {"results": results}
```

Problems with this approach:

**Resource monopolization:** The batch request consumes LLM API quota, memory, and server capacity for its entire duration. Online requests starve.

**No progress visibility:** The client waits with no feedback until everything finishes.

**No partial results:** If the server crashes at item 9,999 of 10,000, all work is lost.

**Timeout risk:** Long requests risk HTTP timeouts, proxy timeouts, client timeouts.

### Job-Based API Design

The correct pattern separates submission from execution:

**Submit endpoint:** Accepts the batch, validates it, creates a job ID, returns immediately.

```python
@app.post("/batch")
async def submit_batch(items: List[str]):
    job_id = create_job(items)
    enqueue_for_processing(job_id)
    return {"job_id": job_id, "status": "accepted"}
```

**Status endpoint:** Returns job progress without blocking.

```python
@app.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    job = get_job(job_id)
    return {
        "status": job.status,
        "progress": job.items_completed,
        "total": job.items_total
    }
```

**Results endpoint:** Returns completed results when ready.

```python
@app.get("/batch/{job_id}/results")
async def get_batch_results(job_id: str):
    job = get_job(job_id)
    if job.status != "completed":
        raise HTTPException(404, "Not ready")
    return {"results": job.results}
```

The batch execution happens in Zone 3 — an external worker, job queue, or background process with its own resources.

### Fairness and Starvation Prevention

Even with job-based design, batch work can starve online traffic if they share downstream resources.

Consider: you have 50 concurrent LLM API slots. If batch processing uses 40 of them, online requests fight over 10.

Solutions:

**Explicit quotas:** Reserve capacity for online traffic. Batch gets "leftover" slots.

**Priority queuing:** Online requests always go first. Batch requests wait.

**Separate pools:** Batch workers use different API keys or endpoints with separate limits.

**Backpressure:** When online load is high, pause batch processing.

The key insight is that fairness must be explicit. The runtime does not automatically balance workloads.

### Mental Image: The Workshop

Imagine your API as a workshop with limited workbenches.

Online requests are customers who walk in and need quick service. They should be helped immediately.

Batch requests are large orders that require hours of work. They should be logged in a notebook and worked on during quiet periods.

If a large order takes over all the workbenches, walk-in customers leave angry.

The solution is not "work faster." It is "separate the queues."

---

## Summary: The Execution Model

You now have a complete picture of how FastAPI applications execute:

**The server is a long-running process.** State persists. Mistakes accumulate. Testing is not production.

**FastAPI is a configuration object.** It declares intent. Uvicorn executes.

**The event loop is the scheduler.** It coordinates work but does not do heavy lifting itself.

**Sync handlers run in thread pools.** Concurrency is bounded by pool size. Exhaustion is graceful but slow.

**Async handlers run on the event loop.** Concurrency is bounded by cooperation. Starvation is sudden and catastrophic.

**Blocking in async code is a bug.** It looks like working code. It fails under load.

**Lifespan owns process resources.** Models, clients, pools. Created once, shared safely.

**Dependencies own request resources.** User context, auth, transactions. Isolated per request.

**Background tasks are not isolated.** They share everything. Heavy work belongs external.

**Batch is not a slow request.** It is a job. Decouple submission from execution.

If you understand these principles, you can reason about FastAPI applications correctly. You can predict what will fail. You can design systems that survive production.

---

## Interview Framing

If an interviewer asks about FastAPI concurrency, here are the key points to hit:

**"How does FastAPI handle concurrent requests?"**

"FastAPI runs on an event loop with a thread pool. Async handlers run as tasks on the event loop and scale well for I/O-bound work as long as they yield at await points. Sync handlers run in a thread pool, which bounds concurrency to the pool size. The critical thing is understanding what blocks what — blocking code in an async handler starves the event loop, which is worse than thread pool exhaustion because it affects all requests at once."

**"Where would you put a large ML model in a FastAPI app?"**

"In the lifespan context, created at startup and stored in app.state. Models are expensive to load and read-only during inference, so they belong at process scope. Creating them per-request would be catastrophically slow, and lazy initialization risks race conditions."

**"How would you handle batch inference in FastAPI?"**

"I wouldn't process it inline. Batch work can run for minutes and would monopolize resources that online requests need. I'd accept the batch, validate it, create a job with an ID, and return immediately. The actual processing happens in an external worker. The client polls a status endpoint or gets a webhook when complete. This separates submission from execution and prevents starvation."
