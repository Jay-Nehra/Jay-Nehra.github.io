# Vector Databases from First Principles

Vector databases are specialized systems for storing and searching high-dimensional vectors. They power the retrieval component of RAG, recommendation systems, image search, and many other applications.

This document explains vector databases from the ground up: what problem they solve, how similarity search works, what indexing strategies exist, and what breaks in production. We'll use Pinecone as a concrete example, but the concepts apply to any vector database.

---

## 1. Why Vector Databases Exist

Vector databases exist because traditional databases cannot efficiently answer the question: "What is most similar to this?"

### The Problem: Semantic Similarity, Not Exact Match

Traditional databases excel at exact matching:
- Find the user with ID 12345
- Select all orders from yesterday
- Get products where category = "electronics"

These are equality or range queries. The database uses indexes (B-trees, hash tables) to find exact matches in O(log n) or O(1) time.

But many applications need similarity:
- Find products similar to this one
- Find documents about this topic
- Find images that look like this
- Find the closest answer to this question

Similarity is not equality. "Similar to" is fuzzy, continuous, and domain-dependent. You cannot hash similarity. You cannot binary search for "close enough."

### Why Traditional Databases Fail

You could store vectors in a traditional database as array columns:

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding FLOAT[1536]
);
```

But searching requires comparing the query vector to every stored vector:

```sql
SELECT id, text, 
       cosine_similarity(embedding, query_vector) as score
FROM documents
ORDER BY score DESC
LIMIT 5;
```

This is O(n) — you scan every row. For a million documents, every query examines a million vectors. For high-dimensional vectors (1536 dimensions), each comparison is expensive.

This doesn't scale. A few thousand documents might be tolerable. Millions become unusable.

### What "Embedding" Actually Means

Before we go further, let's be precise about embeddings.

An embedding is a vector — a fixed-length array of floating-point numbers — that represents some object in a way that captures its meaning. Similar objects have similar embeddings (vectors that are close together in vector space).

For text, embeddings are produced by neural networks trained on massive datasets. The network learns to map text to vectors such that semantically similar texts have similar vectors.

The embedding itself is not interpretable. You cannot look at the 734th dimension and say "this measures sentiment." The dimensions are entangled. What matters is the relative positions of vectors — their distances and angles.

### The Fundamental Operation: Nearest Neighbor Search

The core operation of a vector database is nearest neighbor search:

Given a query vector Q, find the K vectors in the database that are closest to Q.

"Closest" is defined by a distance metric (more on this later). The result is the K most similar items.

This is the retrieval step in RAG. This is how recommendation systems find "items like this one." This is how image search works.

---

## 2. What a Vector Is in This Context

To work with vector databases effectively, you need to understand what vectors actually are and how they behave.

### Fixed-Length Array of Floats

A vector in this context is simply an array of floating-point numbers:

```python
embedding = [0.023, -0.441, 0.872, ..., 0.156]  # 1536 numbers
```

Every vector has the same length (dimensionality). You cannot compare vectors of different dimensions.

The numbers are typically 32-bit floats, sometimes 16-bit for storage efficiency. Each vector takes `dimension * bytes_per_float` storage. A 1536-dimensional vector in float32 takes 6,144 bytes (6KB).

### Dimensionality: 384, 768, 1536

Common embedding dimensions:
- 384 dimensions: Lightweight models (all-MiniLM-L6-v2)
- 768 dimensions: Medium models (all-mpnet-base-v2)
- 1536 dimensions: OpenAI text-embedding-3-small
- 3072 dimensions: OpenAI text-embedding-3-large

Why these numbers? They come from the architecture of the embedding models. 768 is a common transformer hidden size. Multiples and fractions of 768 are typical.

Higher dimensions can capture more nuance but cost more:
- Storage: Linear in dimension
- Search: Roughly linear in dimension (more compute per comparison)
- Memory: Linear in dimension

Diminishing returns apply. Going from 384 to 768 dimensions often helps quality. Going from 1536 to 3072 helps less.

### What Each Dimension "Means"

Nothing. Individually, dimensions are not interpretable.

The embedding model learned to distribute semantic information across all dimensions. No single dimension corresponds to "sentiment" or "topic" or any human-understandable concept.

What matters is the geometry: how vectors relate to each other in the high-dimensional space. Similar texts cluster together. Different topics form different clusters. The structure is in the relationships, not the individual coordinates.

### Why Dimensionality Matters for Storage and Speed

For a database with N vectors of dimension D:

**Storage:** N * D * 4 bytes (for float32)
- 1 million vectors at 1536 dimensions = 6 GB
- 100 million vectors at 1536 dimensions = 600 GB

**Search (brute force):** N * D floating-point operations per query
- 1 million vectors at 1536 dimensions = 1.5 billion operations
- Even at 10 billion ops/second, that's 150ms per query

This is why indexing is essential. Brute force doesn't scale.

---

## 3. Similarity Metrics: The Math That Matters

Distance metrics define what "similar" means. Different metrics give different results.

### Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring their magnitudes.

```
cosine_similarity(A, B) = (A · B) / (|A| * |B|)
```

Where A · B is the dot product and |A| is the magnitude (length) of A.

The result ranges from -1 to 1:
- 1: Identical direction (parallel)
- 0: Perpendicular (unrelated)
- -1: Opposite directions

**When to use:** Text embeddings are typically normalized (magnitude = 1), so cosine similarity is standard. It ignores how "strong" a vector is and focuses on direction.

### Dot Product

The dot product is the cosine similarity multiplied by both magnitudes:

```
dot_product(A, B) = A · B = Σ(A[i] * B[i])
```

For normalized vectors (magnitude 1), dot product equals cosine similarity.

**When to use:** When vectors are already normalized, or when you want magnitude to matter (e.g., more "confident" embeddings should score higher).

### Euclidean Distance

Euclidean distance is the straight-line distance between points:

```
euclidean(A, B) = sqrt(Σ(A[i] - B[i])²)
```

Lower distance = more similar.

**When to use:** When you care about absolute positions in space, not just angles. Common for geographic data or when vectors have meaningful scales.

### When to Use Which

For text embeddings from standard models:
- **Use cosine similarity** — it's the default, handles normalized vectors well
- Models are typically trained with cosine similarity as the objective

For other cases:
- **Normalized vectors:** Cosine and dot product are equivalent; use either
- **Non-normalized vectors where magnitude matters:** Dot product
- **Absolute distances matter:** Euclidean

### What Breaks If You Pick Wrong

Using the wrong metric produces wrong results:

**Euclidean on normalized vectors:** Works, but unnecessarily complex. Euclidean distance and cosine similarity are monotonically related for normalized vectors.

**Dot product on non-normalized vectors:** Results will be biased toward high-magnitude vectors regardless of direction. Longer vectors score higher.

**Mixed normalization:** If some vectors are normalized and some aren't, comparisons are meaningless.

The most common mistake: not realizing your embedding model normalizes (or doesn't). Check documentation.

---

## 4. The Indexing Problem

Brute-force search is O(n). For large databases, this is unacceptable. Indexing makes search fast.

### Brute Force: Why It Fails at Scale

Brute-force search compares the query to every vector:

```python
def brute_force_search(query: Vector, database: List[Vector], k: int):
    scores = []
    for vec in database:
        score = cosine_similarity(query, vec)
        scores.append(score)
    top_k_indices = argsort(scores)[-k:]
    return top_k_indices
```

Time complexity: O(n * d) where n is database size and d is dimension.

For 1 million vectors at 1536 dimensions:
- ~1.5 billion floating-point operations
- At 10 billion ops/sec = 150ms per query
- Multiply by concurrent queries...

At 10 million vectors, queries take seconds. At 100 million, this approach is useless.

### The Tradeoff: Speed vs Accuracy vs Memory

Indexes trade accuracy for speed. Instead of examining all vectors, they examine a subset likely to contain the nearest neighbors.

This is called Approximate Nearest Neighbor (ANN) search.

**Speed:** How fast is search? (queries per second)
**Accuracy:** How often do you find the true nearest neighbors? (recall)
**Memory:** How much RAM does the index require?

You cannot maximize all three. Every index makes tradeoffs:
- More accurate indexes are slower or use more memory
- Faster indexes sacrifice accuracy or require more memory
- Smaller indexes sacrifice speed or accuracy

The art is finding the right balance for your use case.

### Approximate Nearest Neighbor Concept

ANN algorithms work by organizing vectors so that similar vectors are grouped together. When searching:

1. Identify which group(s) the query might belong to
2. Only search within those groups
3. Return the best results found

This is approximate because:
- The true nearest neighbor might be in a group you didn't search
- Grouping is imperfect

With good algorithms and parameters, recall can be 95-99% while searching only 1-10% of the database. 100x speedup for 1-5% accuracy loss is often acceptable.

---

## 5. Indexing Strategies: HNSW

HNSW (Hierarchical Navigable Small World) is the most popular indexing algorithm for vector databases. Understanding it helps you tune parameters effectively.

### How It Works

HNSW builds a multi-layer graph of vectors.

**The graph structure:**
- Each vector is a node
- Nodes are connected to nearby nodes
- Higher layers have fewer nodes and longer-range connections
- Lower layers have more nodes and shorter-range connections

**Building the graph:**
When inserting a vector:
1. Find the nearest neighbors at each layer
2. Add edges from the new vector to those neighbors
3. Randomly decide how many layers to include this vector in

**Searching the graph:**
1. Start at the top layer (sparse, long-range connections)
2. Greedily navigate toward the query vector
3. Drop to lower layers when stuck
4. At the bottom layer, collect the best neighbors found

This is like navigating a city: highways (top layers) get you close quickly, then local streets (bottom layers) find the exact destination.

### Build Time vs Query Time Tradeoffs

HNSW parameters:

**M (edges per node):** More edges = better accuracy but slower builds and more memory.
- Typical: 16-64
- Higher M helps accuracy but has diminishing returns

**ef_construction:** How many candidates to consider during build.
- Higher = better index quality, slower builds
- Typical: 100-500
- Set high if you can afford build time

**ef_search:** How many candidates to consider during search.
- Higher = better accuracy, slower queries
- Can be tuned per query
- Typical: 50-200

### Memory Requirements

HNSW is memory-heavy. The index must fit in RAM.

Memory = vectors + graph edges

For N vectors of dimension D with M edges:
- Vectors: N * D * 4 bytes (float32)
- Graph: N * M * 8 bytes (edges + metadata)

For 10 million vectors at 1536 dimensions with M=16:
- Vectors: 60 GB
- Graph: ~1.3 GB
- Total: ~62 GB RAM needed

This is why HNSW is fast — it trades memory for speed.

### When HNSW Breaks

**High churn:** Deletes and updates are slow. HNSW is optimized for append-only workloads. Heavy deletions degrade index quality.

**Very high dimensions:** As dimensions grow, the "curse of dimensionality" makes distances more uniform. HNSW's graph navigation becomes less effective.

**Memory constraints:** If the index doesn't fit in RAM, performance degrades catastrophically. HNSW is not designed for disk-based operation.

---

## 6. Indexing Strategies: IVF

IVF (Inverted File Index) is an alternative approach that trades some accuracy for reduced memory.

### Clustering Approach

IVF divides the vector space into clusters:

1. **Training:** Run k-means clustering on a sample of vectors. Find N centroids.
2. **Indexing:** For each vector, find its nearest centroid. Store the vector in that centroid's "bucket."
3. **Search:** Find the nearest centroids to the query. Only search vectors in those buckets.

If you search 10 buckets out of 1000, you examine 1% of vectors.

### nprobe Parameter

`nprobe` controls how many clusters to search.

**nprobe = 1:** Only search the nearest cluster. Fast but low recall. The true nearest neighbor might be in an adjacent cluster.

**nprobe = 10:** Search 10 nearest clusters. Slower but much better recall.

**nprobe = all:** Equivalent to brute force.

This is a direct accuracy/speed tradeoff tunable at query time.

### When to Use IVF vs HNSW

**HNSW advantages:**
- Generally better accuracy at same speed
- Good for medium-sized datasets (up to ~100M vectors with enough RAM)
- Better for high recall requirements

**IVF advantages:**
- Lower memory footprint (no graph structure)
- Can use disk-based storage
- Faster index building
- Works with compression (IVF-PQ)

**Use IVF when:**
- Memory is constrained
- Dataset is very large
- You can tolerate lower recall
- You need fast index updates

**Use HNSW when:**
- Dataset fits in RAM
- High recall is important
- Query latency is critical

Many production systems use HNSW for hot data (in RAM) and IVF with compression for cold data (on disk).

---

## 7. Pinecone: Concrete Implementation

Let's make this concrete with Pinecone, a managed vector database. The concepts apply to other systems (Weaviate, Qdrant, Milvus, Chroma).

### Namespace and Index Concepts

**Index:** A collection of vectors, like a database. Each index has a fixed dimension and metric.

**Namespace:** A partition within an index. Vectors in different namespaces are completely isolated. Useful for multi-tenant applications.

```python
import pinecone

# Create index
pinecone.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine"
)

# Connect
index = pinecone.Index("my-index")
```

### Upsert, Query, Delete Operations

**Upsert:** Insert or update vectors.

```python
index.upsert(
    vectors=[
        {
            "id": "doc1",
            "values": [0.1, 0.2, ...],  # 1536 floats
            "metadata": {"source": "report.pdf", "page": 1}
        },
        {
            "id": "doc2",
            "values": [0.3, 0.4, ...],
            "metadata": {"source": "report.pdf", "page": 2}
        }
    ],
    namespace="documents"
)
```

**Query:** Find similar vectors.

```python
results = index.query(
    vector=[0.2, 0.3, ...],  # Query vector
    top_k=5,
    include_metadata=True,
    filter={"source": "report.pdf"},  # Optional metadata filter
    namespace="documents"
)

for match in results.matches:
    print(f"ID: {match.id}, Score: {match.score}")
    print(f"Metadata: {match.metadata}")
```

**Delete:** Remove vectors.

```python
index.delete(
    ids=["doc1", "doc2"],
    namespace="documents"
)

# Or delete by filter
index.delete(
    filter={"source": "old_report.pdf"},
    namespace="documents"
)
```

### Metadata Filtering

Metadata lets you filter results without affecting vector similarity:

```python
# Only return documents from 2024
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={
        "year": {"$gte": 2024}
    }
)

# Complex filters
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={
        "$and": [
            {"category": {"$eq": "documentation"}},
            {"language": {"$in": ["en", "es"]}}
        ]
    }
)
```

Filtering happens before or during vector search (implementation varies). Highly selective filters can speed up search; unselective filters may slow it down.

### What Pinecone Handles vs What You Handle

**Pinecone handles:**
- Index storage and management
- Query routing and load balancing
- Scaling (replicas, sharding)
- Index building and optimization

**You handle:**
- Generating embeddings (Pinecone stores vectors, not text)
- Chunking documents
- Managing metadata
- Application-level logic (RAG pipeline, caching)

### Execution Model of a Query

When you call `index.query()`:

1. **Serialize:** Your query vector and filters are serialized
2. **Network:** Request travels to Pinecone's servers
3. **Routing:** Request is routed to the right shard(s)
4. **Search:** HNSW (or similar) search on each relevant shard
5. **Filter:** Results filtered by metadata
6. **Merge:** Results from shards are merged and ranked
7. **Response:** Top-k results returned to you

Latency depends on:
- Network (typically 5-50ms)
- Index size (larger = slightly slower)
- Filter selectivity (very selective filters can slow down)
- top_k (returning 100 is slower than 5)

Typical production latency: 10-100ms.

---

## 8. Hybrid Search

Pure vector search has limitations. Hybrid search combines vector similarity with keyword matching.

### Why Semantic Search Alone Fails

Vector search finds semantically similar content. But sometimes you need exact matches:

**Technical terms:** "Python 3.12" should match "Python 3.12", not "Python 3.11"

**Names and identifiers:** "John Smith" should match "John Smith", not "Jon Smythe"

**Codes and numbers:** "Error E-7231" should match exactly

**Rare terms:** Words the embedding model rarely saw may not be represented well

Vector search finds "about the same topic." Keyword search finds "contains these words." Both are useful.

### Combining Dense and Sparse Vectors

Hybrid search uses two types of vectors:

**Dense vectors:** Traditional embeddings. Every dimension has a value. Captures semantic meaning.

**Sparse vectors:** Most dimensions are zero. Non-zero dimensions correspond to specific words/tokens. Like TF-IDF or BM25.

```python
# Dense embedding (from model)
dense = [0.1, -0.2, 0.3, ..., 0.05]  # 1536 non-zero values

# Sparse embedding (keyword-based)
sparse = {
    "python": 0.8,    # Word "python" appears, high weight
    "error": 0.5,     # Word "error" appears
    # Most words don't appear -> effectively zero
}
```

Search computes both similarities and combines them:

```python
final_score = alpha * dense_score + (1 - alpha) * sparse_score
```

### Reranking

Another approach: retrieve more candidates than needed, then rerank with a more expensive model.

```python
# Stage 1: Fast vector search, get 50 candidates
candidates = vector_db.query(query_embedding, top_k=50)

# Stage 2: Rerank with cross-encoder
reranked = reranker.rank(query, [c.text for c in candidates])

# Return top 5 after reranking
final = reranked[:5]
```

Cross-encoders (like reranking models) are more accurate than bi-encoders (embedding models) but much slower. They can only be used on small candidate sets.

### Reciprocal Rank Fusion

When combining results from multiple retrieval methods (dense, sparse, keyword), Reciprocal Rank Fusion (RRF) is a simple, effective method:

```python
def rrf_score(ranks: List[int], k: int = 60) -> float:
    """Combine rankings from multiple sources."""
    return sum(1 / (k + rank) for rank in ranks)

# Example: document appears at rank 3 in dense, rank 10 in sparse
score = 1/(60+3) + 1/(60+10)  # Higher is better
```

RRF doesn't require calibrated scores — just ranks. It's robust to different score distributions across methods.

---

## 9. What Breaks in Production

Vector databases have specific failure modes. Understanding them prevents production surprises.

### Dimensionality Mismatch Bugs

The most common bug: vectors have the wrong dimension.

```python
# Index expects 1536 dimensions
# You send 768 dimensions
index.upsert([{"id": "doc1", "values": embedding_768}])
# Error! Dimension mismatch
```

This happens when:
- You change embedding models without rebuilding the index
- You mix embeddings from different models
- You truncate embeddings without updating the index

Prevention:
- Validate dimensions before upserting
- Include embedding model name in index name
- Test after any model changes

### Index Rebuild Costs and Downtime

Changing embedding models requires rebuilding the index. This means:
- Re-embed all documents (compute cost, time)
- Re-index everything (time)
- Potential downtime during transition

For large indexes, this can take hours or days.

Migration strategies:
- **Blue-green:** Build new index while old one serves traffic. Switch when ready.
- **Dual-write:** Write to both indexes during transition.
- **Incremental:** If possible, update gradually (rarely possible with model changes).

Plan for this before you need it.

### Stale Embeddings

Documents change. Embeddings become stale.

If a document is updated but not re-embedded:
- Search finds the old content
- User sees new content
- Mismatch confuses users

Synchronization approaches:
- **Trigger-based:** Re-embed on document update
- **Periodic:** Batch re-embed on schedule
- **Hybrid:** Trigger for important changes, periodic for others

Track embedding timestamps. Alert on staleness.

### Cost at Scale

Vector databases can be expensive:

**Storage:** Vectors are large. 10M vectors at 1536 dimensions = 60GB just for vectors, plus metadata and index overhead.

**Compute:** Every query uses CPU/GPU for similarity computation. High query volume = high compute cost.

**Managed service pricing:** Pinecone, Weaviate Cloud, etc. charge for storage + compute. Costs can surprise you at scale.

Cost optimization:
- Lower dimensions if quality permits
- Fewer vectors (better chunking, deduplication)
- Cache frequent queries
- Batch queries where possible
- Consider self-hosting for very large scale

---

## Summary: The Vector Database Mental Model

Vector databases store high-dimensional vectors and find nearest neighbors efficiently.

**Vectors** are fixed-length arrays of floats. Dimension affects storage, speed, and quality.

**Similarity metrics** define "close." Cosine similarity is standard for text embeddings.

**Indexing** makes search fast. HNSW trades memory for speed with high accuracy. IVF uses less memory but lower accuracy.

**Hybrid search** combines semantic (dense) and keyword (sparse) matching. Neither alone is sufficient.

**Production challenges** include dimension mismatches, index rebuilds, stale embeddings, and costs.

---

## Interview Framing

**"How do vector databases work?"**

"Vector databases store embeddings — high-dimensional vectors that represent semantic meaning — and enable fast similarity search. They use approximate nearest neighbor algorithms like HNSW to avoid O(n) brute-force search. HNSW builds a navigable graph structure that lets you find similar vectors by searching a small fraction of the database. The key tradeoffs are between speed, accuracy, and memory."

**"When would you use hybrid search?"**

"Pure vector search finds semantically similar content but can miss exact matches — specific names, codes, technical terms. Hybrid search combines dense embeddings with sparse keyword vectors like BM25. You'd use it when precision matters for specific terms, like searching documentation where 'Python 3.12' should match exactly, or when users search with product IDs and names."

**"What are the main challenges with vector databases in production?"**

"Dimensionality mismatch when you change embedding models — the new embeddings are incompatible with the old index. Index rebuilds are expensive and potentially cause downtime. Stale embeddings when documents are updated but not re-embedded. And cost scales with storage and query volume, which can surprise you. You need synchronization strategies, blue-green deployments for model changes, and cost monitoring."
