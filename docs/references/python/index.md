# Python Internals

Deep reference guides for Python's runtime behavior, concurrency models, and internal mechanisms. These documents go beyond syntax to explain what is actually happening when our code runs, why Python is designed this way, and what breaks when we misuse these features.

Each guide is written for someone who knows basic Python syntax but wants to understand the machinery underneath. We focus on practical patterns for building LLM applications, APIs, and data pipelines.

---

## Start Here: Understanding Any Object

- [**Introspection and Protocols**](introspection-and-protocols.md) - How to use `dir()` to understand any object, Python protocols and what dunder method combinations enable

---

## Concurrency and Parallelism

How Python handles concurrent and parallel execution, and when to use each approach.

- [**GIL and Threading**](gil-and-threading.md) - The Global Interpreter Lock, when it matters, and the complete threading module walkthrough
- [**Multiprocessing**](multiprocessing.md) - Process-based parallelism for CPU-bound work, inter-process communication
- [**Async Execution Model**](async-execution-model.md) - Event loops, coroutines, and practical async patterns for I/O-bound work

---

## Memory and Object Lifecycle

How Python manages memory and object creation/destruction.

- [**Memory Management**](memory-management.md) - Reference counting, garbage collection, debugging memory leaks

---

## Functions and Code Organization

How Python handles functions, scope, and module loading.

- [**Decorators and Closures**](decorators-and-closures.md) - First-class functions, closure mechanics, practical decorator patterns
- [**Generators and Iteration**](generators-and-iteration.md) - Lazy evaluation, iterator protocol, streaming patterns
- [**Import System**](import-system.md) - Module loading mechanics, project structure, avoiding circular imports
- [**Context Managers**](context-managers.md) - Resource management with `with`, custom context managers

---

## Common Pitfalls

Classic Python gotchas that appear in interviews and cause bugs in production.

- [**Common Gotchas**](common-gotchas.md) - Mutable defaults, is vs ==, late binding, and other traps
