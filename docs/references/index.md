# References

Systems-level understanding of Python libraries, APIs, and AI systems. Not syntax tutorials—these are deep dives into **what's actually happening** under the hood.

## Philosophy

Each guide answers the questions tutorials skip:

- What happens when we call this function?
- Who calls whom, and when?
- Why is it designed this way?
- What breaks if we misunderstand?

## Structure

Every guide has two parts:

1. **Architecture** — The mental model, the machinery, the "why"
2. **Scenarios** — End-to-end examples that exercise the full system

---

## Browse by Category

### Python Internals
- [Introspection and Protocols](python/introspection-and-protocols.md) — Using `dir()` to understand any object, Python protocols and dunder methods
- [GIL and Threading](python/gil-and-threading.md) — The Global Interpreter Lock, threading module, and concurrent I/O patterns
- [Multiprocessing](python/multiprocessing.md) — Process-based parallelism for CPU-bound work
- [Async Execution Model](python/async-execution-model.md) — Event loops, coroutines, and practical async patterns
- [Memory Management](python/memory-management.md) — Reference counting, garbage collection, and memory profiling
- [Decorators and Closures](python/decorators-and-closures.md) — First-class functions, closure mechanics, and practical decorators
- [Generators and Iteration](python/generators-and-iteration.md) — Lazy evaluation, iterator protocol, and streaming patterns
- [Import System](python/import-system.md) — Module loading, project structure, and circular imports
- [Context Managers](python/context-managers.md) — Resource management with the `with` statement
- [Common Gotchas](python/common-gotchas.md) — Classic pitfalls and interview favorites

### API & Web Servers
- [FastAPI Execution Model](api/fastapi-execution-model.md) — How requests actually execute, sync vs async, concurrency, and resource management

### LLM Systems
- [LLM API Execution Model](llm/api-patterns.md) — Tokens, streaming, error handling, context windows, and function calling
- [RAG Architecture](llm/rag-architecture.md) — Retrieval-Augmented Generation from first principles
- [Agent Architecture](llm/agent-architecture.md) — The agent loop, tool calling, state management, and multi-agent patterns

### Databases
- [Vector Databases](databases/vector-databases.md) — Embeddings, similarity search, indexing strategies (HNSW, IVF), and Pinecone

### Data Engineering
- [Text Data Engineering](data/text-data-engineering.md) — Making messy text usable for LLM pipelines with polars
- [Data Profiling](data/data-profiling.md) — Understanding data before transformation: schema, distributions, missing values
- [Data Cleaning](data/data-cleaning.md) — Type coercion, normalization, and systematic transformation
- [Quality Validation](data/quality-validation.md) — Rule-based and statistical validation for data pipelines
- [Correction Strategies](data/correction-strategies.md) — When to fix, impute, quarantine, or reject data

### AI Agents (Framework-Specific)
- [Strands Agents](agents/strands-agents.md) — Complete systems architecture for building production AI agents with Strands SDK

### Configuration
- [Environment Variables & dotenv](config/environment.md) — The foundation for secrets and config

### Serialization
- [JSON Module](serialization/json.md) — What serialization actually means

### Filesystem
- [pathlib](filesystem/pathlib.md) — Modern path handling

### HTTP
- [httpx](http/httpx.md) — Sync and async HTTP client

### CLI
- [argparse](cli/argparse.md) — Command-line argument parsing

---

*These are personal reference notes for systems-level understanding.*
