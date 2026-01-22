# RAG Architecture from First Principles

Retrieval-Augmented Generation (RAG) is a pattern for giving LLMs access to knowledge they weren't trained on. This document explains RAG from the ground up: why it exists, how each component works, and what breaks in production.

Understanding RAG deeply means understanding that it is not a library or a tool — it is an execution pipeline with distinct phases, each with its own failure modes and optimization opportunities.

---

## 1. Why RAG Exists

RAG exists because LLMs have fundamental knowledge limitations that cannot be solved by the models themselves.

### The Knowledge Cutoff Problem

Every LLM is trained on data up to a specific point in time. GPT-4 might have a knowledge cutoff of April 2023. Claude might know about events up to January 2024. Anything that happened after the cutoff is unknown to the model.

This is not a bug or a limitation that can be trained away. It is inherent to how models are created. Training takes months and costs millions of dollars. Models cannot be continuously retrained with yesterday's news.

For any application that needs current information — stock prices, recent events, updated documentation, your company's internal data — the model's training data is insufficient.

### The Private Knowledge Problem

Even if you could continuously retrain, there's another problem: private knowledge.

Your company's internal documents, customer data, proprietary research — none of this was in the training data. The model has never seen it and cannot answer questions about it.

You could fine-tune the model on your data, but this is:
- Expensive (thousands to millions of dollars)
- Slow (weeks to months)
- Inflexible (can't easily update)
- Potentially insecure (your data becomes part of the model)

RAG offers an alternative: keep your data in a database, retrieve relevant pieces at query time, and inject them into the prompt.

### Fine-Tuning vs Retrieval

Fine-tuning and retrieval solve different problems:

**Fine-tuning** teaches the model new behaviors, styles, or formats. It changes how the model responds. Use it for:
- Custom response styles
- Domain-specific terminology
- Following specific formats
- Adjusting tone and personality

**Retrieval** gives the model access to facts it doesn't know. It changes what the model knows for a specific query. Use it for:
- Current information
- Private/proprietary data
- Large document collections
- Frequently updated knowledge

Most applications need retrieval, not fine-tuning. Fine-tuning is for changing behavior; RAG is for adding knowledge.

### The Fundamental Insight: Context Is Computation

Here's the key insight that makes RAG work:

LLMs can effectively use information placed in their context window. If you put relevant text in the prompt, the model can reason about it, synthesize it, and answer questions about it.

This means you can "teach" the model anything at query time by putting it in the context. No training required. The model reads the context, understands it, and uses it to generate a response.

RAG is simply: (1) figure out what context is relevant, and (2) put it in the prompt.

---

## 2. The RAG Pipeline as Execution Flow

RAG is a pipeline with distinct phases. Understanding each phase helps you debug problems and optimize performance.

### Query → Embed → Retrieve → Augment → Generate

The complete flow for a RAG query:

1. **Query arrives:** User asks a question
2. **Query embedding:** Convert the question to a vector representation
3. **Retrieval:** Search your knowledge base for relevant documents
4. **Augmentation:** Construct a prompt with retrieved context
5. **Generation:** Send to LLM, receive response

Each phase has latency, cost, and failure modes.

### What Happens at Each Step

**Query embedding:** Your question ("How do I reset my password?") is passed through an embedding model. This produces a vector — a list of numbers that represents the semantic meaning of the question. This typically takes 10-50ms.

**Retrieval:** The query vector is compared against pre-computed document vectors in a vector database. The database returns the top-k most similar documents. This typically takes 10-100ms depending on database size and infrastructure.

**Augmentation:** You construct a prompt that includes:
- A system message explaining the task
- The retrieved documents as context
- The user's original question
This is just string formatting — essentially free.

**Generation:** The augmented prompt is sent to the LLM. The model reads the context and generates a response. This is the slowest and most expensive step, typically 1-30 seconds.

### Where Latency Lives

Total RAG latency is dominated by generation. A typical breakdown:
- Query embedding: 20ms
- Vector search: 50ms  
- Prompt construction: 1ms
- LLM generation: 2000-5000ms

For optimization, focus on generation first. Reducing context size reduces generation time. Caching repeated queries avoids generation entirely.

### Where Failures Happen

Each phase can fail:

**Embedding:** The embedding model might be unavailable or rate-limited.

**Retrieval:** The query might not match relevant documents (semantic gap). The documents might not contain the answer. The vector database might be slow or unavailable.

**Augmentation:** Context might exceed token limits. Prompt might be malformed.

**Generation:** LLM might be unavailable, rate-limited, or produce low-quality output. Model might ignore context and hallucinate.

Understanding where failure occurs is essential for debugging. A wrong answer could be a retrieval failure (wrong documents retrieved) or a generation failure (right documents, wrong interpretation).

---

## 3. Document Ingestion: The Offline Pipeline

Before you can retrieve documents, you must process and index them. This happens offline, before any queries.

### Loading Documents

Documents come in many formats:
- Plain text
- Markdown
- PDF
- Word documents
- HTML
- Code files

Each format requires different parsing. PDFs are particularly challenging — they're designed for visual display, not text extraction. OCR may be needed for scanned documents.

The goal of loading is to extract clean text that preserves the document's meaning. This is harder than it sounds:
- Headers and footers appear on every page
- Tables lose structure when converted to text
- Images and diagrams are lost
- Multi-column layouts scramble reading order

Garbage in, garbage out. Bad document parsing produces bad retrieval.

### Chunking: Why It Exists

Documents are typically too large to fit in context. A 50-page PDF might be 30,000 tokens. You can't pass the whole thing to the LLM for every query.

Chunking breaks documents into smaller pieces. Each chunk is embedded and indexed separately. At query time, you retrieve individual chunks, not whole documents.

Chunking serves two purposes:
1. **Fit in context:** Chunks are small enough to include multiple in a single prompt
2. **Improve retrieval precision:** Smaller chunks can match more specific queries

### Chunk Size Tradeoffs

Chunk size is one of the most important parameters in RAG.

**Too small (< 200 tokens):**
- Individual chunks lack context
- "The meeting" — which meeting?
- More chunks means more noise in retrieval
- Higher chance of breaking mid-sentence or mid-thought

**Too large (> 1000 tokens):**
- Chunks contain multiple topics
- Retrieval becomes less precise
- Fewer chunks fit in context
- Pay for irrelevant content

**Typical range:** 300-800 tokens

The right size depends on your documents and queries. Technical documentation might need larger chunks to preserve code blocks. FAQs might work well with smaller chunks.

### Overlap and Its Purpose

Chunks typically overlap — the end of one chunk is the beginning of the next.

Why overlap?

Without overlap, a relevant sentence might be split between chunks. Neither chunk contains the complete thought. With overlap, the sentence appears in full in at least one chunk.

Typical overlap: 10-20% of chunk size. A 500-token chunk might overlap by 50-100 tokens.

More overlap means more redundancy (higher storage, slower indexing) but better retrieval coverage.

### Metadata Extraction

Each chunk should carry metadata:
- Source document (title, URL, file path)
- Position in document (page number, section heading)
- Timestamps (creation date, last modified)
- Custom fields (author, category, access level)

Metadata enables:
- Filtering retrieval (only search documents from 2024)
- Attribution in responses ("According to the Q3 report...")
- Access control (only show documents user can access)

Don't skip metadata. It's essential for production systems.

---

## 4. Chunking Strategies Deep Dive

Chunking is not just "split by character count." Different strategies produce dramatically different results.

### Fixed-Size Chunking

The simplest approach: split text every N characters or tokens.

```python
def fixed_size_chunk(text: str, chunk_size: int = 1000, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Pros:**
- Simple to implement
- Predictable chunk sizes
- Works for any text

**Cons:**
- Ignores document structure
- Cuts mid-sentence, mid-paragraph, mid-word
- Semantic units are broken arbitrarily

This is often "good enough" for prototypes but rarely optimal.

### Semantic Chunking

Split at natural boundaries: sentences, paragraphs, or sections.

```python
def sentence_chunk(text: str, max_tokens: int = 500):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

**Pros:**
- Preserves sentence integrity
- More coherent chunks
- Better for natural language documents

**Cons:**
- Chunk sizes vary more
- Very long sentences cause problems
- Doesn't respect higher-level structure

### Recursive Chunking

Split by document structure, falling back to smaller units as needed.

```python
def recursive_chunk(text: str, max_tokens: int = 500, separators: list = None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    
    if count_tokens(text) <= max_tokens:
        return [text]
    
    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            for part in parts:
                chunks.extend(recursive_chunk(part, max_tokens, separators[1:]))
            return chunks
    
    # Fallback: character split
    return fixed_size_chunk(text, max_tokens)
```

**Pros:**
- Respects document structure (sections, paragraphs)
- Falls back gracefully
- Works well for structured documents (Markdown, documentation)

**Cons:**
- More complex
- Behavior depends on document format
- May produce very small chunks for odd documents

### Parent-Child Chunking

Store small chunks for retrieval but return larger context.

The idea: embed small, precise chunks (200 tokens) for accurate matching. But when returning context, include the parent chunk (1000 tokens) for completeness.

```python
# During indexing
for document in documents:
    large_chunks = split_into_large_chunks(document, size=1000)
    for i, large_chunk in enumerate(large_chunks):
        small_chunks = split_into_small_chunks(large_chunk, size=200)
        for j, small_chunk in enumerate(small_chunks):
            store({
                "id": f"{doc_id}_{i}_{j}",
                "text": small_chunk,
                "embedding": embed(small_chunk),
                "parent_text": large_chunk,
                "parent_id": f"{doc_id}_{i}"
            })

# During retrieval
results = vector_db.search(query_embedding, top_k=5)
# Return parent_text instead of text
contexts = [r.parent_text for r in results]
```

**Pros:**
- Precise retrieval + complete context
- Best of both worlds

**Cons:**
- More storage
- More complex indexing
- Must deduplicate parents

### What Breaks with Wrong Chunking

Chunking failures are common and hard to diagnose:

**Chunks too small:** "When was the product launched?" retrieves "launched in 2023" but lacks product name. Model can't answer.

**Chunks too large:** Multiple products in one chunk. Query retrieves chunk but model cites wrong product.

**Mid-thought breaks:** Chunk ends with "The solution is to" — next chunk has the solution but wasn't retrieved.

**Code blocks split:** Function definition in one chunk, body in another. Neither is useful alone.

Test chunking with real queries before going to production.

---

## 5. Embedding Models

Embedding models convert text to vectors. Understanding how they work helps you make better choices.

### What Embedding Models Actually Do

An embedding model is a neural network trained to map text to a fixed-size vector of floating-point numbers. Similar texts produce similar vectors (close in vector space). Dissimilar texts produce distant vectors.

The training objective is usually "contrastive": push similar pairs together, push dissimilar pairs apart. Models are trained on massive datasets of text pairs, learning general semantic similarity.

### Sentence Transformers vs OpenAI Embeddings

Two main options:

**Sentence Transformers (open source):**
- Run locally or self-hosted
- No API costs
- Lower latency (no network)
- Full control
- Many models, varying quality
- Popular: all-MiniLM-L6-v2 (384 dim), all-mpnet-base-v2 (768 dim)

**OpenAI/Commercial embeddings:**
- Hosted API
- Per-token pricing
- Higher latency (network)
- Generally higher quality for general text
- Less control
- Popular: text-embedding-3-small (1536 dim), text-embedding-3-large (3072 dim)

For most applications, either works. Choose based on:
- Cost sensitivity (self-host = cheaper at scale)
- Quality requirements (commercial often better for general text)
- Latency requirements (local = faster)
- Operational complexity (API = simpler)

### Dimension vs Quality Tradeoffs

Embedding dimension affects:

**Storage:** 1536-dim vectors use 4x the storage of 384-dim vectors.

**Search speed:** Higher dimensions slow search, especially brute-force.

**Quality:** More dimensions can capture more nuance, but diminishing returns apply.

Modern models often support "Matryoshka" embeddings — you can truncate the vector to fewer dimensions with graceful degradation. This enables tuning the dimension/quality tradeoff at query time.

### Batch Embedding for Performance

Embedding is parallelizable. For ingestion, batch documents:

```python
# Bad: one document at a time
for doc in documents:
    embedding = embed(doc)  # One API call per document
    store(doc, embedding)

# Good: batch processing
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    embeddings = embed_batch(batch)  # One API call per batch
    for doc, embedding in zip(batch, embeddings):
        store(doc, embedding)
```

This dramatically reduces:
- Total time (parallelism)
- API calls (relevant for hosted models)
- Network overhead

### Model Choice: When It Matters

Embedding model choice matters most when:

**Domain-specific content:** General models may miss domain terminology. Consider domain-specific models or fine-tuning.

**Multiple languages:** Some models are English-only. Check language support.

**Code:** Code embeddings need code-aware models. General text models often fail.

**Very short queries:** Some models are optimized for short queries vs. long documents.

**Asymmetric search:** Query style differs from document style (questions vs. answers). Some models handle this better.

When in doubt, test. Create a small benchmark of queries and relevant documents, then measure retrieval quality with different models.

---

## 6. Retrieval: The Query Path

When a query arrives, you must find the most relevant documents. This is where the quality of your RAG system is often determined.

### Query Embedding

The query is embedded using the same model used for documents. This is essential — different models produce incompatible embeddings.

Query embedding is usually fast (10-50ms for a short query). It's also cacheable — identical queries can reuse embeddings.

```python
query = "How do I reset my password?"
query_embedding = embed(query)  # Same model as documents
```

### Top-K Retrieval

The query embedding is compared against all document embeddings. The most similar documents are returned.

```python
results = vector_db.search(
    vector=query_embedding,
    top_k=5,  # Return 5 most similar
    filter={"category": "documentation"}  # Optional metadata filter
)
```

The "top-k" parameter controls how many results to return. Common values are 3-10.

More results mean:
- Higher chance of including the relevant document
- More tokens in context (higher cost)
- More noise (irrelevant documents)

Fewer results mean:
- Lower cost
- Higher risk of missing relevant content
- Less noise (if retrieval is good)

### Similarity Thresholds vs Fixed K

Two approaches to filtering results:

**Fixed K:** Always return exactly K results.
- Predictable token count
- May return irrelevant results if no good matches
- May miss relevant results if K is too small

**Similarity threshold:** Return results above a similarity score.
- Variable result count
- Avoids low-quality matches
- May return nothing for unusual queries
- Threshold tuning is difficult

In practice, combining both often works well: return up to K results, but only those above a minimum similarity threshold.

```python
results = vector_db.search(
    vector=query_embedding,
    top_k=10,  # Maximum results
)
# Filter by score
results = [r for r in results if r.score > 0.7]
# Limit to 5
results = results[:5]
```

### What "Relevant" Means (and Doesn't)

Vector similarity measures semantic similarity — whether two texts are "about the same thing."

Similarity does NOT measure:
- **Correctness:** A document might be semantically similar but factually wrong
- **Answer presence:** A document about the topic might not contain the answer
- **Recency:** Old and new documents on the same topic are equally similar
- **Authority:** All sources are equal in vector space

High similarity means "related topic." It does not mean "good answer."

This is why retrieval is necessary but not sufficient. You still need the LLM to interpret and synthesize the retrieved content.

---

## 7. Context Assembly

After retrieval, you must construct a prompt for the LLM. This step is often underestimated.

### Ordering Retrieved Chunks

Retrieved chunks can be ordered by:
- Similarity score (most similar first)
- Document order (original sequence)
- Recency (newest first)
- Custom ranking

Order matters because of position bias in LLMs. Some models pay more attention to content at the beginning or end of the context, less to the middle (the "lost in the middle" problem).

If you suspect position bias, put the most relevant content first and last.

### Deduplication

If you use parent-child chunking or overlapping chunks, retrieved content may be redundant. The same sentence might appear in multiple chunks.

Deduplicate before constructing context:

```python
def deduplicate_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    unique = []
    for chunk in chunks:
        # Simple approach: dedupe by exact match
        if chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
    return unique
```

For overlapping chunks, you might merge overlapping text regions rather than simple deduplication.

### Token Budget Management

Your context window is limited. Budget tokens across:
- System prompt (usually fixed)
- Retrieved context (variable)
- Conversation history (variable)
- Reserved for output (must leave room)

```python
MAX_CONTEXT = 8000
SYSTEM_PROMPT_TOKENS = 500
MAX_OUTPUT_TOKENS = 1000
MAX_RETRIEVED_TOKENS = MAX_CONTEXT - SYSTEM_PROMPT_TOKENS - MAX_OUTPUT_TOKENS

def fit_chunks_to_budget(chunks: List[str], budget: int) -> List[str]:
    result = []
    used = 0
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if used + chunk_tokens > budget:
            break
        result.append(chunk)
        used += chunk_tokens
    return result
```

If you exceed the context, either truncate or summarize. Never send requests that exceed limits.

### The "Lost in the Middle" Problem

Research shows that LLMs often pay more attention to content at the beginning and end of long contexts, with reduced attention to content in the middle.

For RAG, this means:
- Highly relevant content in the middle may be ignored
- The model might answer from less-relevant content at edges
- Very long contexts can actually reduce quality

Mitigations:
- Keep contexts shorter (5-10 chunks, not 50)
- Put most relevant content first
- Consider reranking retrieved chunks
- Use models specifically optimized for long contexts

---

## 8. Generation with Context

The final step: sending the assembled prompt to the LLM and getting a response.

### System Prompt Design for RAG

Your system prompt should:
1. Explain the model's role
2. Describe how to use the provided context
3. Set expectations for handling missing information

Example:

```
You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the user's question using ONLY the information in the context below.
- If the context does not contain the answer, say "I don't have information about that in the provided documents."
- When possible, cite which document your answer came from.
- Do not make up information that is not in the context.

Context:
{retrieved_documents}

User question: {query}
```

Be explicit. Don't assume the model will figure out what you want.

### Grounding the Model in Retrieved Content

The goal of RAG is grounded generation — answers based on your documents, not the model's training.

But models can still:
- Hallucinate additional details
- Combine context with training knowledge
- Ignore context entirely for confident answers

Reinforce grounding:
- Explicitly instruct "only use the provided context"
- Ask for citations
- Use lower temperature (less creativity)
- Validate outputs against retrieved content

### Citation and Attribution

For trustworthiness, have the model cite sources:

```
Based on the Q3 Financial Report, revenue increased by 15%...
```

You can enforce this structurally:

```python
prompt = """
Answer the question based on the context. For each fact in your answer, cite the source in [brackets].

Context:
[Source: Q3 Report] Revenue increased 15% year over year.
[Source: Press Release] The new product launched in September.

Question: What happened in Q3?
Answer:
"""
```

Or ask for structured output:

```python
response_schema = {
    "answer": "string",
    "sources": ["list of source names used"]
}
```

### Handling "I Don't Know"

The model should refuse to answer when context is insufficient.

This is difficult because:
- Models are trained to be helpful
- Models have training knowledge they might use
- Models sometimes hallucinate rather than admit ignorance

Strategies:
- Explicit instruction: "If you don't know, say 'I don't know'"
- Two-step: First ask "Can you answer based on the context?" then answer
- Confidence scoring: Have model rate its confidence
- Retrieval quality check: Don't send to LLM if retrieval failed

There's no perfect solution. Test with known-unknowable questions and tune prompts.

---

## 9. What Breaks in Production

RAG systems fail in predictable ways. Understanding these failures helps you build more robust systems.

### Query-Document Mismatch

Users ask questions differently than documents are written.

**User:** "How do I fix the thing that's not working?"  
**Document:** "Troubleshooting Error Code E-7231"

The user's vague query doesn't match the document's specific terminology. Vector similarity may not bridge this gap.

Mitigations:
- Query expansion (add synonyms, rephrase)
- Hypothetical Document Embedding (HyDE) — generate what an answer might look like, embed that
- Hybrid search (keyword + semantic)
- Better document preprocessing (add summaries, keywords)

### Retrieval Failures

The relevant document exists but wasn't retrieved.

Causes:
- Query too different from document phrasing
- Document chunk split the answer
- Not enough top-k results
- Wrong embedding model for domain

Diagnosis: Log retrieved documents and manually check for failures. Build evaluation datasets.

### Hallucination Despite Context

The model has the right context but still hallucinates.

Causes:
- Model's training knowledge contradicts context
- Context is unclear or ambiguous
- Instructions not strong enough
- Temperature too high

Mitigations:
- Stronger grounding instructions
- Lower temperature
- Output validation
- Citation requirements

### Latency at Scale

RAG adds latency at every step:
- Query embedding
- Vector search
- Context assembly
- Longer prompts = longer generation

At scale, each step needs optimization:
- Cache embeddings
- Use faster vector databases
- Limit context size
- Stream responses

### Evaluation: How to Know If It's Working

RAG quality is hard to measure because it depends on:
- Retrieval quality (did you get the right documents?)
- Generation quality (did the model use them correctly?)

Evaluation approaches:

**Retrieval metrics:** Measure if relevant documents are in top-k results. Requires labeled query-document pairs.

**End-to-end metrics:** Compare generated answers to gold-standard answers. Requires labeled query-answer pairs.

**LLM-as-judge:** Have another LLM evaluate answer quality. Cheaper than human evaluation but noisy.

**Human evaluation:** Best quality signal but expensive and slow.

For production, combine:
- Automated metrics (retrieval recall, answer similarity)
- Sampled human evaluation
- User feedback (thumbs up/down, corrections)

---

## Summary: The RAG Mental Model

RAG is a pipeline that retrieves relevant documents and uses them as context for LLM generation.

**Ingestion** happens offline: load documents, chunk them, embed chunks, store in vector database.

**Query** happens at request time: embed query, search for similar chunks, assemble context, generate response.

**Chunking** is critical. Too small loses context. Too large loses precision. Test with real queries.

**Retrieval** finds related documents, not correct answers. Similarity != answer quality.

**Generation** must be grounded. Instruct the model explicitly. Require citations. Handle unknowns.

**Production** adds complexity: evaluation, caching, latency optimization, failure handling.

---

## Interview Framing

**"What is RAG and when would you use it?"**

"RAG — Retrieval-Augmented Generation — is a pattern for giving LLMs access to knowledge they weren't trained on. You'd use it when you need current information, private data, or any knowledge not in the model's training set. It's an alternative to fine-tuning that's more flexible and easier to update. The core idea is: retrieve relevant documents at query time and include them in the prompt as context."

**"How would you design a RAG system for company documentation?"**

"First, ingest: parse documents, chunk them with overlap for context preservation, embed with a model like OpenAI's embeddings or sentence-transformers, store in a vector database like Pinecone. At query time: embed the query, retrieve top-5 similar chunks, assemble them into a prompt that instructs the model to answer only from context, and stream the response. Key decisions are chunk size — I'd test 400-600 tokens — and whether to use hybrid search for keyword matching on technical terms."

**"What are the main failure modes in RAG?"**

"Retrieval failures are most common: the relevant document exists but wasn't retrieved because the query phrasing was too different. Chunking failures: the answer was split across chunks and neither was complete. Grounding failures: the model ignores context and uses training knowledge or hallucinates. For each, you need different fixes: query expansion for retrieval, overlap and parent-child chunking for splits, stronger prompting and lower temperature for grounding."
