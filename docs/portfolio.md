# Portfolio

This page highlights selected systems and projects I’ve worked on over the years.
Rather than listing roles or responsibilities, it focuses on the kinds of systems I’ve built and the problems they were designed to solve.

---

## Job Standardization & Classification Systems

I’ve worked on rebuilding and maintaining large-scale job standardization pipelines used to map external job data to evolving internal job architectures.

These systems typically involve:
- noisy and inconsistent training data
- evolving taxonomies and business rules
- high accuracy and stability requirements across multiple downstream products

A significant part of this work focused on large-scale data cleanup and validation, improving model behavior and prediction robustness. The pipelines covered training, inference, post-processing, evaluation, and production monitoring.

**Themes:** data quality, ML evaluation, production reliability.

---

## Embedding Models & Evaluation Infrastructure

I’ve designed and maintained configuration-driven frameworks for evaluating embedding models across multiple tasks, including:
- clustering
- classification
- retrieval

These frameworks enabled repeatable benchmarking, automated metrics, and side-by-side comparisons to support research-driven model selection. The goal was to make model evaluation systematic rather than ad hoc.

**Themes:** reproducibility, experimentation, ML platform design.

---

## LLM-Powered Applications & RAG Pipelines

Across multiple projects, I’ve built LLM-powered applications that allow users to interact with structured and unstructured data through natural language.

Examples include:
- natural-language-to-SQL analytics assistants
- document-grounded chat systems
- domain-specific conversational tools

These systems relied on retrieval-augmented generation (RAG) and vector search to ground model responses in proprietary datasets, improving factual accuracy and relevance.

**Themes:** LLM orchestration, retrieval systems, AI application design.

---

## Agentic AI & Workflow Automation

More recently, I’ve worked on agentic AI systems designed to orchestrate workflows across internal APIs and services.

This included:
- designing MCP servers
- implementing AI agents for natural-language-driven workflows
- integrating human-in-the-loop controls for reliability and oversight

The focus has been on building systems that augment human workflows rather than fully automating them.

**Themes:** agent design, orchestration, system boundaries.

---

## Data Platforms & Pipelines

I’ve built and maintained analytical and ML-focused data platforms using Databricks, BigQuery, and cloud-native tooling.

These platforms supported:
- analytics and reporting
- ML experimentation and training
- production inference pipelines

The work involved data modeling, pipeline design, monitoring, and performance optimization, with an emphasis on maintainability over time.

**Themes:** data engineering, scalability, long-lived systems.

---

## Earlier Systems Work

Earlier in my career, I worked on backend services, event-driven pipelines, and infrastructure components, including:
- real-time ingestion using Kafka and Apache Pulsar
- graph-based data modeling with Neo4j
- containerized microservices and observability tooling

This foundation strongly influences how I approach reliability and system design today.

---

## What I care about

Across projects, a few recurring interests show up:
- building systems that survive change
- making evaluation a first-class concern
- reducing hidden complexity
- writing code and documentation that future engineers can understand
