"""
Mass EU generation for EpiChat — targets 5000+ EpistemicUnits.

Three-pronged approach:
  1. Wikipedia scraping: ~300 topics × 25 sentences ≈ 2000 new EUs (free, fast)
  2. LLM synthesis:      Ollama llama3.1:8b generates structured EUs for
                         1000+ specific programming concepts not well-covered
                         by Wikipedia articles (e.g. "when to use a deque",
                         "common React pitfalls", "Python GIL explained")
  3. Stack Overflow API: Top questions by tag → high-confidence EUs from
                         accepted answers (optional, requires network)

Run overnight:
    source /home/me/SSM/.venv/bin/activate
    python scripts/generate_eus.py 2>&1 | tee logs/eu_expansion.log
"""

import json
import os
import re
import sys
import time
import urllib.request

EPICHAT_DIR = "/home/me/EpiChat"
sys.path.insert(0, EPICHAT_DIR)

from core.knowledge_graph import KnowledgeGraph
from core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from seeding.code_seeder import CodeSeeder


# ─── 1. Extended Wikipedia topic list (~300 topics) ──────────────────────────

WIKI_TOPICS = [
    # Core CS
    "Algorithm", "Data structure", "Big O notation", "Sorting algorithm",
    "Binary search algorithm", "Dynamic programming", "Graph theory", "Hash table",
    "Tree (data structure)", "Linked list", "Stack (abstract data type)",
    "Queue (abstract data type)", "Heap (data structure)", "Trie",
    "Red–black tree", "AVL tree", "B-tree", "Segment tree", "Fenwick tree",
    "Skip list", "Bloom filter", "Disjoint-set data structure",
    "Topological sorting", "Strongly connected component", "Minimum spanning tree",
    "Dijkstra's algorithm", "A* search algorithm", "Bellman–Ford algorithm",
    "Floyd–Warshall algorithm", "Greedy algorithm", "Divide-and-conquer algorithm",
    "Backtracking", "Randomized algorithm", "Amortized analysis",
    "String-searching algorithm", "Suffix array", "Suffix tree",
    "Network flow", "Maximum flow problem", "Linear programming",
    "NP-completeness", "P versus NP problem",

    # OOP & Patterns
    "Object-oriented programming", "Encapsulation (computer programming)",
    "Inheritance (object-oriented programming)", "Polymorphism (computer science)",
    "SOLID", "Software design pattern", "Singleton pattern", "Observer pattern",
    "Factory method pattern", "Abstract factory pattern", "Builder pattern",
    "Decorator pattern", "Adapter pattern", "Command pattern", "Strategy pattern",
    "Template method pattern", "Proxy pattern", "Composite pattern",
    "Iterator pattern", "Facade pattern", "Flyweight pattern",
    "Chain-of-responsibility pattern", "Mediator pattern", "Memento pattern",
    "State pattern", "Visitor pattern", "Bridge pattern",
    "Dependency injection", "Inversion of control",

    # ML & AI
    "Machine learning", "Deep learning", "Neural network (machine learning)",
    "Convolutional neural network", "Recurrent neural network",
    "Transformer (deep learning architecture)", "Attention (machine learning)",
    "Backpropagation", "Gradient descent", "Stochastic gradient descent",
    "Overfitting", "Regularization (mathematics)", "Cross-validation (statistics)",
    "Decision tree", "Random forest", "Gradient boosting",
    "Support vector machine", "K-nearest neighbors algorithm",
    "Naive Bayes classifier", "Logistic regression", "Linear regression",
    "Reinforcement learning", "Q-learning", "Natural language processing",
    "Word embedding", "Word2vec", "BERT (language model)",
    "Generative adversarial network", "Variational autoencoder",
    "Transfer learning", "Batch normalization", "Dropout (neural networks)",
    "Long short-term memory", "Gated recurrent unit",
    "Principal component analysis", "K-means clustering",
    "Autoencoder", "Diffusion model",

    # Databases
    "Database", "Relational database", "SQL", "NoSQL",
    "Database index", "Database normalization", "ACID",
    "Database transaction", "Query optimization", "PostgreSQL",
    "MongoDB", "Redis", "Elasticsearch", "CAP theorem",
    "Eventual consistency", "Database sharding", "Replication (computing)",
    "Two-phase commit protocol", "Write-ahead logging", "MVCC",
    "Object–relational mapping", "Stored procedure", "Trigger (databases)",
    "Database view", "Materialized view", "Apache Cassandra",
    "InfluxDB", "Graph database", "Neo4j",

    # Distributed Systems
    "Distributed computing", "Consensus (computer science)", "Raft (algorithm)",
    "Paxos (computer science)", "MapReduce", "Apache Kafka",
    "Message queue", "Remote procedure call", "gRPC",
    "Distributed hash table", "Consistent hashing",
    "Saga pattern", "Two-phase commit protocol",
    "Byzantine fault tolerance", "Leader election",
    "Service discovery", "Apache Zookeeper",

    # Systems & OS
    "Operating system", "Process (computing)", "Thread (computing)",
    "Concurrency (computer science)", "Deadlock", "Semaphore (programming)",
    "Mutex", "Memory management", "Virtual memory", "Cache (computing)",
    "CPU cache", "Garbage collection (computer science)", "Compiler",
    "Just-in-time compilation", "Interpreter (computing)",
    "Abstract syntax tree", "Parsing", "Lexical analysis",
    "Context switch", "Scheduling (computing)", "Memory paging",
    "Copy-on-write", "Memory-mapped file", "I/O bound",
    "CPU-bound", "NUMA", "False sharing",

    # Languages & Paradigms
    "Python (programming language)", "Kotlin (programming language)",
    "Java (programming language)", "JavaScript", "TypeScript",
    "Rust (programming language)", "Go (programming language)",
    "C++ (programming language)", "Functional programming",
    "Type system", "Type inference", "Generic programming",
    "Metaprogramming", "Reflection (computer programming)",
    "Continuation-passing style", "Monads (functional programming)",
    "Lazy evaluation", "Currying", "Higher-order function",
    "Closure (computer programming)", "First-class function",
    "Async/await", "Coroutine", "Event loop",

    # Web & Cloud
    "REST", "GraphQL", "WebSocket", "HTTP",
    "Hypertext Transfer Protocol Secure", "OAuth", "JSON Web Token",
    "Microservices", "Service mesh", "API gateway",
    "Docker (software)", "Kubernetes", "Continuous integration",
    "Continuous deployment", "Infrastructure as code",
    "Serverless computing", "Cloud computing",
    "Content delivery network", "Load balancing (computing)",
    "Circuit breaker (computing)", "Rate limiting",

    # Security
    "Computer security", "OWASP", "SQL injection",
    "Cross-site scripting", "Cross-site request forgery",
    "Public-key cryptography", "Symmetric-key algorithm",
    "Transport Layer Security", "Authentication", "Authorization",
    "Penetration testing", "Firewall (computing)",
    "Buffer overflow", "Race condition", "Zero-day vulnerability",
    "Intrusion detection system", "Security information and event management",
    "Hash function", "SHA-2", "Bcrypt", "Salting (cryptography)",

    # Architecture
    "Software architecture", "Model–view–controller", "CQRS",
    "Event-driven architecture", "Hexagonal architecture",
    "Domain-driven design", "Event sourcing",
    "Strangler fig pattern", "Sidecar pattern",

    # Testing
    "Test-driven development", "Unit testing", "Integration testing",
    "Behavior-driven development", "Property-based testing",
    "Fuzzing", "Mutation testing", "Code coverage",
    "Test doubles", "Mocking",

    # Tools & Practices
    "Version control", "Git", "Continuous integration",
    "Refactoring", "Technical debt", "Code review",
    "Pair programming", "Code smell", "Clean code",
    "Software metric", "Cyclomatic complexity",
]


# ─── 2. LLM-synthesized EU topics ─────────────────────────────────────────────

# Programming concepts that need structured EU format with code, complexity,
# tradeoffs — better synthesized via LLM than scraped from Wikipedia
LLM_CONCEPTS = [
    # Python-specific
    ("Python list comprehension vs generator expression", "python"),
    ("Python __slots__ for memory optimization", "python"),
    ("Python descriptors and the descriptor protocol", "python"),
    ("Python context managers and the with statement", "python"),
    ("Python asyncio event loop internals", "python"),
    ("Python GIL and multiprocessing vs threading", "python"),
    ("Python dataclasses vs namedtuple vs TypedDict", "python"),
    ("Python type hints and mypy static analysis", "python"),
    ("Python functools.lru_cache and caching decorators", "python"),
    ("Python pathlib vs os.path for file operations", "python"),
    ("Python logging best practices", "python"),
    ("Python virtual environments and dependency management", "python"),
    ("Python property decorator for computed attributes", "python"),
    ("Python metaclasses and class creation", "python"),
    ("Python __init__ vs __new__ vs __post_init__", "python"),
    ("Python walrus operator := and its use cases", "python"),
    ("Python match statement for structural pattern matching", "python"),
    ("Python collections.defaultdict vs dict.setdefault", "python"),
    ("Python itertools for lazy sequence operations", "python"),
    ("Python heapq for priority queues", "python"),

    # Data structures - practical
    ("When to use a deque vs list in Python", "data_structures"),
    ("Circular buffer implementation and use cases", "data_structures"),
    ("LRU cache implementation with OrderedDict or doubly linked list", "data_structures"),
    ("Monotonic stack for next greater element problems", "data_structures"),
    ("Union-Find with path compression and rank optimization", "data_structures"),
    ("Segment tree with lazy propagation", "data_structures"),
    ("Persistent data structures and immutability", "data_structures"),
    ("Rope data structure for string editing", "data_structures"),
    ("Van Emde Boas tree for integer sets", "data_structures"),
    ("Treap: randomized BST combining tree and heap", "data_structures"),
    ("Splay tree and self-adjusting BSTs", "data_structures"),
    ("Interval tree for overlapping interval queries", "data_structures"),
    ("2D range tree and orthogonal range search", "data_structures"),
    ("Adjacency list vs adjacency matrix for graphs", "data_structures"),
    ("Hash collision resolution: chaining vs open addressing", "data_structures"),

    # Algorithm patterns
    ("Sliding window technique for subarray problems", "algorithms"),
    ("Two pointers technique for sorted array problems", "algorithms"),
    ("Fast and slow pointers for cycle detection", "algorithms"),
    ("Merge intervals pattern", "algorithms"),
    ("Cyclic sort for missing/duplicate number problems", "algorithms"),
    ("BFS for shortest path in unweighted graphs", "algorithms"),
    ("DFS for topological sort and cycle detection", "algorithms"),
    ("Bidirectional BFS for word ladder problems", "algorithms"),
    ("A* search with admissible heuristic", "algorithms"),
    ("Dijkstra's algorithm with priority queue", "algorithms"),
    ("Bellman-Ford for negative weight edges", "algorithms"),
    ("Floyd-Warshall for all-pairs shortest paths", "algorithms"),
    ("Kruskal's algorithm for MST with Union-Find", "algorithms"),
    ("Prim's algorithm for MST with priority queue", "algorithms"),
    ("Knapsack 0/1 dynamic programming", "algorithms"),
    ("Longest common subsequence dynamic programming", "algorithms"),
    ("Longest increasing subsequence with patience sorting", "algorithms"),
    ("Matrix chain multiplication DP", "algorithms"),
    ("Edit distance (Levenshtein) DP", "algorithms"),
    ("KMP string matching algorithm", "algorithms"),
    ("Z-algorithm for pattern matching", "algorithms"),
    ("Rabin-Karp rolling hash for pattern matching", "algorithms"),
    ("Manacher's algorithm for longest palindromic substring", "algorithms"),
    ("Boyer-Moore voting algorithm for majority element", "algorithms"),
    ("Reservoir sampling for streaming data", "algorithms"),
    ("Fisher-Yates shuffle for random permutation", "algorithms"),
    ("Quickselect for kth smallest element", "algorithms"),
    ("Counting sort and radix sort for linear time", "algorithms"),
    ("Sieve of Eratosthenes for prime generation", "algorithms"),
    ("Modular exponentiation for large powers", "algorithms"),
    ("Extended Euclidean algorithm and GCD", "algorithms"),
    ("Miller-Rabin primality test", "algorithms"),
    ("Convex hull algorithms: Graham scan and Jarvis march", "algorithms"),
    ("Line sweep algorithm for geometric problems", "algorithms"),
    ("Fenwick tree for range sum queries with point updates", "algorithms"),
    ("Segment tree for range min/max/sum queries", "algorithms"),

    # Concurrency
    ("Race condition prevention with locks and atomics", "concurrency"),
    ("Deadlock prevention: lock ordering and timeout", "concurrency"),
    ("Producer-consumer pattern with bounded buffer", "concurrency"),
    ("Thread pool executor for CPU-bound tasks", "concurrency"),
    ("Asyncio coroutines for I/O-bound concurrency", "concurrency"),
    ("Lock-free programming with compare-and-swap", "concurrency"),
    ("Read-write lock for concurrent reads", "concurrency"),
    ("Semaphore for resource counting", "concurrency"),
    ("Barrier synchronization for parallel phases", "concurrency"),
    ("Actor model for message-passing concurrency", "concurrency"),
    ("Software transactional memory", "concurrency"),
    ("Futures and promises for async composition", "concurrency"),

    # Database patterns
    ("Index selectivity and query optimization", "databases"),
    ("Composite index vs covering index vs partial index", "databases"),
    ("N+1 query problem and eager loading solutions", "databases"),
    ("Connection pooling best practices", "databases"),
    ("Optimistic vs pessimistic locking", "databases"),
    ("Database cursor for large result sets", "databases"),
    ("Upsert patterns: INSERT ... ON CONFLICT", "databases"),
    ("Soft delete vs hard delete trade-offs", "databases"),
    ("Temporal tables for audit trails", "databases"),
    ("Full-text search with PostgreSQL tsvector", "databases"),
    ("JSONB in PostgreSQL for semi-structured data", "databases"),
    ("Redis data types: strings, hashes, lists, sets, sorted sets", "databases"),
    ("Redis pub/sub vs streams for messaging", "databases"),
    ("Time-series data modeling in InfluxDB and TimescaleDB", "databases"),
    ("Graph database query patterns with Cypher", "databases"),

    # API design
    ("REST API versioning strategies", "architecture"),
    ("Idempotent HTTP operations and retry safety", "architecture"),
    ("Pagination: cursor-based vs offset-based", "architecture"),
    ("API rate limiting: token bucket and sliding window", "architecture"),
    ("OpenAPI spec-first API design", "architecture"),
    ("gRPC vs REST vs GraphQL trade-offs", "architecture"),
    ("Webhook design and delivery guarantees", "architecture"),
    ("API authentication: API keys vs OAuth2 vs JWT", "architecture"),
    ("Hypermedia APIs and HATEOAS", "architecture"),

    # Distributed systems patterns
    ("Circuit breaker pattern for fault tolerance", "architecture"),
    ("Bulkhead pattern for resource isolation", "architecture"),
    ("Saga pattern for distributed transactions", "architecture"),
    ("Outbox pattern for reliable message publishing", "architecture"),
    ("CQRS: separating read and write models", "architecture"),
    ("Event sourcing: state from events", "architecture"),
    ("Eventual consistency and conflict resolution", "architecture"),
    ("Consistent hashing for load distribution", "architecture"),
    ("Leader election with Raft consensus", "architecture"),
    ("Two-phase commit vs saga for distributed transactions", "architecture"),
    ("Write-ahead log for durability", "architecture"),
    ("Read-your-writes consistency guarantee", "architecture"),
    ("Chaos engineering principles and practices", "architecture"),

    # Security
    ("OWASP Top 10: injection flaws and prevention", "cybersecurity"),
    ("SQL injection prevention with parameterized queries", "cybersecurity"),
    ("XSS prevention: output encoding and CSP headers", "cybersecurity"),
    ("CSRF prevention with SameSite cookies and tokens", "cybersecurity"),
    ("Password hashing with bcrypt vs argon2", "cybersecurity"),
    ("JWT validation: signature, expiry, and claims", "cybersecurity"),
    ("OAuth2 authorization code flow with PKCE", "cybersecurity"),
    ("Secrets management: never hardcode credentials", "cybersecurity"),
    ("TLS mutual authentication (mTLS)", "cybersecurity"),
    ("Container security: least privilege and image scanning", "cybersecurity"),
    ("Dependency vulnerability scanning with tools", "cybersecurity"),
    ("Input validation vs output encoding distinction", "cybersecurity"),

    # Testing patterns
    ("Test pyramid: unit, integration, e2e ratio", "testing"),
    ("Property-based testing with Hypothesis", "testing"),
    ("Parameterized tests for covering edge cases", "testing"),
    ("Test fixtures and factory patterns", "testing"),
    ("Mocking external dependencies for isolation", "testing"),
    ("Contract testing with Pact for microservices", "testing"),
    ("Mutation testing to validate test quality", "testing"),
    ("Performance testing: load, stress, soak tests", "testing"),
    ("Snapshot testing for UI components", "testing"),
    ("Test double patterns: stub, spy, mock, fake", "testing"),

    # Code quality
    ("SOLID principles applied to Python code", "best_practices"),
    ("DRY vs WET: when repetition is acceptable", "best_practices"),
    ("Premature optimization and profiling first", "best_practices"),
    ("Composition over inheritance in practice", "best_practices"),
    ("Feature flags for safe deployments", "best_practices"),
    ("Trunk-based development vs feature branches", "best_practices"),
    ("Semantic versioning for library releases", "best_practices"),
    ("Documentation as code with docstrings and ADRs", "best_practices"),
    ("Error handling: fail fast vs defensive programming", "best_practices"),
    ("Structured logging with correlation IDs", "best_practices"),
    ("Observability: metrics, logs, and traces", "best_practices"),
    ("Code review best practices and pull request hygiene", "best_practices"),
    ("Refactoring: extract method, extract class, move method", "best_practices"),
    ("Technical debt: types and management strategies", "best_practices"),

    # Performance
    ("CPU cache optimization: data locality and struct layout", "performance"),
    ("Memory allocator internals and fragmentation", "performance"),
    ("SIMD instructions for data-parallel computation", "performance"),
    ("Branch prediction and avoiding branch mispredictions", "performance"),
    ("I/O patterns: buffered, direct, memory-mapped", "performance"),
    ("Connection pooling and keep-alive for HTTP clients", "performance"),
    ("Lazy loading vs eager loading trade-offs", "performance"),
    ("Compression algorithms: LZ4, Snappy, Zstandard", "performance"),
    ("Profiling Python with cProfile and py-spy", "performance"),
    ("Database query explain plan interpretation", "performance"),
    ("Index scan vs full table scan in query planning", "performance"),
    ("Caching strategies: write-through, write-back, write-around", "performance"),
    ("CDN edge caching and cache invalidation strategies", "performance"),
    ("HTTP/2 multiplexing and header compression", "performance"),
]

EU_GENERATION_PROMPT = """You are a structured knowledge base for software engineering.

Generate a concise, accurate EpistemicUnit for this programming concept:
"{concept}"

Respond in EXACTLY this JSON format (no other text):
{{
  "proposition": "One clear sentence describing what this concept IS or DOES (30-100 words)",
  "time_complexity": "Big-O time complexity if applicable, else null",
  "space_complexity": "Big-O space complexity if applicable, else null",
  "when_to_use": "2-3 sentence practical guidance on when to apply this (null if not applicable)",
  "tradeoffs": "Key advantages and disadvantages (null if not applicable)",
  "code_snippet": "Minimal Python code example (10-30 lines) demonstrating the concept, else null",
  "confidence": 0.85
}}"""


def call_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 60) -> str | None:
    """Call Ollama REST API to generate a response."""
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except Exception as e:
        print(f"  [Ollama error] {e}", file=sys.stderr)
        return None


def parse_eu_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def check_ollama_running() -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


EPICHAT_DATA = os.path.join(EPICHAT_DIR, "episteme_data")


def main():
    kg = KnowledgeGraph()
    kg.load(EPICHAT_DATA)
    print(f"Starting EU count: {len(kg.units)}")
    print(f"Target: 5000+ EUs\n")

    total_added = 0

    # ── Phase 1: Wikipedia scraping (~300 topics) ─────────────────────────────
    print(f"=== Phase 1: Wikipedia scraping ({len(WIKI_TOPICS)} topics) ===")
    seeder = CodeSeeder(kg)
    n = seeder.seed_wikipedia(topics=WIKI_TOPICS, max_sentences=25)
    total_added += n
    print(f"  Added {n} EUs from Wikipedia. Total: {len(kg.units)}\n")
    kg.save(EPICHAT_DATA)

    # ── Phase 2: LLM synthesis ────────────────────────────────────────────────
    print(f"=== Phase 2: LLM synthesis ({len(LLM_CONCEPTS)} concepts) ===")

    ollama_ok = check_ollama_running()
    if not ollama_ok:
        print("  [WARN] Ollama not running. Starting...")
        os.system("ollama serve > /tmp/ollama.log 2>&1 &")
        time.sleep(5)
        ollama_ok = check_ollama_running()

    if not ollama_ok:
        print("  [SKIP] Ollama not available — skipping LLM synthesis")
    else:
        # Pick best available model
        model = "qwen35-teacher" if _model_exists("qwen35-teacher") else "llama3.1:8b"
        print(f"  Using model: {model}")

        source = Source(name=f"LLM-synthesized ({model})", reliability_score=0.80)
        n_llm = 0
        n_skip = 0

        for i, (concept, domain) in enumerate(LLM_CONCEPTS):
            prompt = EU_GENERATION_PROMPT.format(concept=concept)
            raw = call_ollama(prompt, model=model)
            if not raw:
                n_skip += 1
                continue

            parsed = parse_eu_json(raw)
            if not parsed or not parsed.get("proposition"):
                n_skip += 1
                continue

            eu = EpistemicUnit(
                proposition=parsed["proposition"],
                knowledge_type=KnowledgeType.BEST_PRACTICE,
                confidence=float(parsed.get("confidence", 0.80)),
                domain=domain,
                sources=[source],
                code_snippet=parsed.get("code_snippet"),
                time_complexity=parsed.get("time_complexity"),
                space_complexity=parsed.get("space_complexity"),
                when_to_use=parsed.get("when_to_use"),
                tradeoffs=parsed.get("tradeoffs"),
            )

            if kg.add(eu):
                n_llm += 1
                total_added += 1
                print(f"  [{i+1:03d}/{len(LLM_CONCEPTS)}] +1 | {concept[:60]}")
            else:
                n_skip += 1
                print(f"  [{i+1:03d}/{len(LLM_CONCEPTS)}] dup | {concept[:60]}")

            # Save every 50 new EUs
            if n_llm > 0 and n_llm % 50 == 0:
                kg.save(EPICHAT_DATA)
                print(f"  [SAVE] {len(kg.units)} total EUs saved")

        print(f"\n  LLM synthesis: {n_llm} added, {n_skip} skipped")

    # ── Final save & export ───────────────────────────────────────────────────
    print(f"\n=== Final save ===")
    kg.save(EPICHAT_DATA)
    print(f"Total EU count: {len(kg.units)} (added {total_added} in this run)")

    print("\n=== Re-exporting training traces ===")
    os.system(
        "cd /home/me/SSM && source .venv/bin/activate 2>/dev/null; "
        "python -u -m train.epichat_export "
        "--epichat-dir /home/me/EpiChat "
        "--output data/epichat_traces.jsonl "
        "--min-confidence 0.4"
    )
    print("Done.")


def _model_exists(name: str) -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return any(m["name"] == name or m["name"].startswith(name + ":")
                       for m in data.get("models", []))
    except Exception:
        return False


if __name__ == "__main__":
    main()
