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
