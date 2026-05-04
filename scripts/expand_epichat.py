"""
Expand EpiChat knowledge graph from ~10k to ~100k units.

Steps:
  1. Load existing KG from episteme_data/
  2. Seed 800+ Wikipedia CS articles (max_sentences=50 each)
  3. Seed 500+ hand-crafted EXTRA_AXIOMS
  4. Save KG (units.json, graph.pkl, faiss.index, faiss_map.json)
  5. Rebuild FAISS index from scratch with sentence-transformers
  6. Upload epichat/{units.json,faiss.index,faiss_map.json} to HF dataset

Run:
    cd /home/me/SSM
    source .venv/bin/activate
    python scripts/expand_epichat.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
# EpiChat lives at /home/me/EpiChat — insert so "from core.xxx import …" works.
EPICHAT_ROOT = "/home/me/EpiChat"
if EPICHAT_ROOT not in sys.path:
    sys.path.insert(0, EPICHAT_ROOT)

EPISTEME_DATA = os.path.join(EPICHAT_ROOT, "episteme_data")
HF_REPO       = "pgalyen1987/rs-code-ssm-traces"

# ── Imports (after path setup) ────────────────────────────────────────────────
from core.knowledge_graph import KnowledgeGraph
from core.epistemic_unit  import EpistemicUnit, KnowledgeType, Source
from seeding.code_seeder  import CodeSeeder


# ═══════════════════════════════════════════════════════════════════════════════
# 800+ Wikipedia article titles
# ═══════════════════════════════════════════════════════════════════════════════

WIKI_TOPICS = [
    # ── Sorting algorithms ──────────────────────────────────────────────────
    "Quicksort",
    "Merge sort",
    "Heapsort",
    "Timsort",
    "Radix sort",
    "Counting sort",
    "Bucket sort",
    "Insertion sort",
    "Selection sort",
    "Bubble sort",
    "Shell sort",
    "Introsort",
    "Cycle sort",
    "Patience sorting",
    "Smoothsort",

    # ── Searching ───────────────────────────────────────────────────────────
    "Binary search algorithm",
    "Interpolation search",
    "Exponential search",
    "Jump search",
    "Fibonacci search technique",
    "Ternary search",
    "Linear search",

    # ── Graph algorithms ────────────────────────────────────────────────────
    "Dijkstra's algorithm",
    "Bellman–Ford algorithm",
    "Floyd–Warshall algorithm",
    "A* search algorithm",
    "Breadth-first search",
    "Depth-first search",
    "Topological sorting",
    "Kruskal's algorithm",
    "Prim's algorithm",
    "Strongly connected component",
    "Tarjan's strongly connected components algorithm",
    "Kosaraju's algorithm",
    "Johnson's algorithm",
    "Borůvka's algorithm",
    "Edmonds–Karp algorithm",
    "Ford–Fulkerson algorithm",
    "Maximum flow problem",
    "Minimum spanning tree",
    "Travelling salesman problem",
    "Graph coloring",
    "Bipartite graph",
    "Eulerian path",
    "Hamiltonian path",

    # ── Dynamic programming ─────────────────────────────────────────────────
    "Dynamic programming",
    "Knapsack problem",
    "Longest common subsequence",
    "Longest increasing subsequence",
    "Edit distance",
    "Matrix chain multiplication",
    "Coin problem",
    "Fibonacci sequence",
    "Memoization",
    "Optimal substructure",
    "Overlapping subproblems",

    # ── String algorithms ───────────────────────────────────────────────────
    "Knuth–Morris–Pratt algorithm",
    "Rabin–Karp algorithm",
    "Z function (computer science)",
    "Suffix array",
    "Aho–Corasick algorithm",
    "Levenshtein distance",
    "Longest common substring",
    "String-searching algorithm",
    "Regular expression",
    "Boyer–Moore string-search algorithm",

    # ── Computational geometry ──────────────────────────────────────────────
    "Computational geometry",
    "Convex hull",
    "Graham scan",
    "Jarvis march",
    "Closest pair of points problem",
    "Line segment intersection",
    "Polygon triangulation",
    "Sweep line algorithm",

    # ── Numerical methods ───────────────────────────────────────────────────
    "Numerical analysis",
    "Newton's method",
    "Bisection method",
    "Runge–Kutta methods",
    "Gaussian elimination",
    "LU decomposition",
    "Fast Fourier transform",
    "Discrete Fourier transform",
    "Numerical integration",

    # ── Data structures ─────────────────────────────────────────────────────
    "Array data structure",
    "Linked list",
    "Doubly linked list",
    "Stack (abstract data type)",
    "Queue (abstract data type)",
    "Double-ended queue",
    "Priority queue",
    "Binary tree",
    "Binary search tree",
    "AVL tree",
    "Red–black tree",
    "B-tree",
    "B+ tree",
    "Heap (data structure)",
    "Fibonacci heap",
    "Binomial heap",
    "Hash table",
    "Trie",
    "Suffix tree",
    "Segment tree",
    "Fenwick tree",
    "Disjoint-set data structure",
    "Skip list",
    "Bloom filter",
    "LRU cache",
    "Sparse table",
    "Van Emde Boas tree",
    "Treap",
    "Splay tree",
    "Rope (data structure)",
    "Interval tree",
    "K-d tree",
    "Quadtree",
    "Octree",
    "Cartesian tree",
    "Left-leaning red–black tree",
    "Count–min sketch",
    "HyperLogLog",

    # ── GoF Design Patterns ─────────────────────────────────────────────────
    "Abstract factory pattern",
    "Builder pattern",
    "Factory method pattern",
    "Prototype pattern",
    "Singleton pattern",
    "Adapter pattern",
    "Bridge pattern",
    "Composite pattern",
    "Decorator pattern",
    "Facade pattern",
    "Flyweight pattern",
    "Proxy pattern",
    "Chain-of-responsibility pattern",
    "Command pattern",
    "Interpreter pattern",
    "Iterator pattern",
    "Mediator pattern",
    "Memento pattern",
    "Observer pattern",
    "State pattern",
    "Strategy pattern",
    "Template method pattern",
    "Visitor pattern",
    "Design Patterns (book)",
    "Software design pattern",

    # ── Enterprise / architectural patterns ─────────────────────────────────
    "Repository pattern",
    "Unit of work pattern",
    "CQRS",
    "Event sourcing",
    "Saga pattern",
    "Outbox pattern",
    "Domain-driven design",
    "Hexagonal architecture",
    "Clean architecture",
    "Model–view–controller",
    "Model–view–presenter",
    "Model–view–viewmodel",
    "Dependency injection",
    "Inversion of control",
    "Service locator pattern",
    "Microservices",
    "Service-oriented architecture",
    "Monolithic application",
    "Twelve-factor app",
    "Event-driven architecture",

    # ── Programming languages ───────────────────────────────────────────────
    "Python (programming language)",
    "JavaScript",
    "TypeScript",
    "Java (programming language)",
    "Go (programming language)",
    "Rust (programming language)",
    "C++ (programming language)",
    "C (programming language)",
    "Kotlin (programming language)",
    "Swift (programming language)",
    "Scala (programming language)",
    "Haskell (programming language)",
    "Erlang (programming language)",
    "Elixir (programming language)",
    "Clojure",
    "Ruby (programming language)",
    "PHP",
    "R (programming language)",
    "Julia (programming language)",
    "Dart (programming language)",
    "Lua (programming language)",
    "Racket (programming language)",
    "OCaml",
    "F Sharp (programming language)",
    "Lisp (programming language)",
    "Prolog",
    "Assembly language",
    "WebAssembly",

    # ── Concurrency & parallelism ────────────────────────────────────────────
    "Concurrent computing",
    "Parallel computing",
    "Thread (computing)",
    "Mutex",
    "Semaphore (programming)",
    "Deadlock",
    "Race condition",
    "Lock (computer science)",
    "Monitor (synchronization)",
    "Actor model",
    "Coroutine",
    "Async/await",
    "Event loop",
    "Futures and promises",
    "Memory model (programming)",
    "Volatile (computer programming)",
    "Compare-and-swap",
    "Spinlock",
    "Readers–writer lock",
    "Producer–consumer problem",
    "Dining philosophers problem",
    "Communicating sequential processes",
    "Software transactional memory",
    "Lock-free data structure",
    "Transactional memory",
    "OpenMP",
    "SIMD",
    "Thread pool",
    "Work stealing",
    "Amdahl's law",
    "Gustafson's law",

    # ── Databases ────────────────────────────────────────────────────────────
    "Relational database",
    "SQL",
    "NoSQL",
    "ACID",
    "Database normalization",
    "Database index",
    "B-tree index",
    "Query optimization",
    "Transaction processing",
    "Database sharding",
    "Replication (computing)",
    "CAP theorem",
    "Eventual consistency",
    "PostgreSQL",
    "MySQL",
    "SQLite",
    "MongoDB",
    "Redis",
    "Apache Cassandra",
    "Apache Kafka",
    "Elasticsearch",
    "Graph database",
    "Time series database",
    "Column-oriented DBMS",
    "Data warehouse",
    "Extract, transform, load",
    "OLAP",
    "OLTP",
    "Database connection pooling",
    "Optimistic concurrency control",
    "Two-phase locking",
    "Multiversion concurrency control",
    "Database trigger",
    "Stored procedure",
    "View (SQL)",
    "Window function (SQL)",
    "CockroachDB",
    "Apache Spark",
    "Apache Hadoop",
    "ClickHouse",
    "DuckDB",

    # ── Networking ───────────────────────────────────────────────────────────
    "HTTP",
    "HTTPS",
    "HTTP/2",
    "HTTP/3",
    "WebSocket",
    "Representational state transfer",
    "GRPC",
    "GraphQL",
    "Transmission Control Protocol",
    "User Datagram Protocol",
    "IP address",
    "Domain Name System",
    "Transport Layer Security",
    "OAuth",
    "JSON Web Token",
    "API gateway",
    "Load balancing (computing)",
    "Reverse proxy",
    "Content delivery network",
    "Network address translation",
    "Firewall (computing)",
    "Virtual private network",
    "Microservices",
    "Service mesh",
    "Circuit breaker (computing)",
    "Rate limiting",
    "OSI model",
    "Internet protocol suite",
    "Anycast",
    "BGP",
    "QUIC",
    "Server-sent events",
    "Long polling",
    "Zero-copy",
    "CORS",
    "Same-origin policy",

    # ── Software engineering ──────────────────────────────────────────────────
    "Software testing",
    "Unit testing",
    "Integration testing",
    "Test-driven development",
    "Behavior-driven development",
    "Code coverage",
    "Mock object",
    "Continuous integration",
    "Continuous delivery",
    "DevOps",
    "Agile software development",
    "Scrum (software development)",
    "Kanban (development)",
    "Technical debt",
    "Code smell",
    "Refactoring",
    "Software architecture",
    "Clean code",
    "Pair programming",
    "Code review",
    "Version control",
    "Git",
    "Software metric",
    "Cyclomatic complexity",
    "Static program analysis",
    "Software documentation",
    "API design",
    "Semantic versioning",
    "Feature flag",
    "A/B testing",
    "Blue-green deployment",
    "Canary release",
    "Software testing anti-patterns",
    "Mutation testing",
    "Property-based testing",
    "Fuzzing",
    "Contract testing",

    # ── Security ──────────────────────────────────────────────────────────────
    "Computer security",
    "Cryptography",
    "Public-key cryptography",
    "Symmetric-key algorithm",
    "Cryptographic hash function",
    "Digital signature",
    "Certificate authority",
    "OAuth",
    "OpenID Connect",
    "Cross-origin resource sharing",
    "Cross-site scripting",
    "Cross-site request forgery",
    "SQL injection",
    "OWASP",
    "Penetration testing",
    "Vulnerability (computing)",
    "Zero-day (computing)",
    "Buffer overflow",
    "Memory safety",
    "Fuzzing",
    "Static program analysis",
    "Privilege escalation",
    "Injection attack",
    "Side-channel attack",
    "Timing attack",
    "AES (cryptography)",
    "RSA (cryptosystem)",
    "Elliptic-curve cryptography",
    "SHA-2",
    "bcrypt",
    "Key derivation function",
    "TLS handshake",
    "Perfect forward secrecy",
    "Secure coding",
    "Defense in depth (computing)",

    # ── Math / CS theory ─────────────────────────────────────────────────────
    "Computational complexity theory",
    "NP-completeness",
    "P versus NP problem",
    "Automata theory",
    "Formal language",
    "Turing machine",
    "Lambda calculus",
    "Type theory",
    "Category theory",
    "Linear algebra",
    "Probability theory",
    "Information theory",
    "Entropy (information theory)",
    "Bayesian inference",
    "Monte Carlo method",
    "Gradient descent",
    "Backpropagation",
    "Big O notation",
    "Amortized analysis",
    "Recurrence relation",
    "Master theorem (analysis of algorithms)",
    "Halting problem",
    "Undecidable problem",
    "Pumping lemma",
    "Context-free grammar",
    "Chomsky hierarchy",

    # ── Cloud / infrastructure ────────────────────────────────────────────────
    "Cloud computing",
    "Serverless computing",
    "Containerization",
    "Docker (software)",
    "Kubernetes",
    "Infrastructure as code",
    "Terraform (software)",
    "CI/CD",
    "Blue-green deployment",
    "Canary release",
    "Chaos engineering",
    "Observability (software)",
    "Distributed tracing",
    "Log management",
    "Application performance management",
    "Site reliability engineering",
    "Service-level objective",
    "Service-level agreement",
    "Function as a service",
    "Platform as a service",
    "Infrastructure as a service",
    "Cloud-native computing",
    "Helm (package manager)",
    "Istio (software)",
    "Envoy (software)",
    "Prometheus (software)",
    "Grafana",
    "OpenTelemetry",

    # ── Compiler / language internals ─────────────────────────────────────────
    "Compiler",
    "Lexical analysis",
    "Parsing",
    "Abstract syntax tree",
    "Semantic analysis (compilers)",
    "Intermediate representation",
    "Just-in-time compilation",
    "Ahead-of-time compilation",
    "Garbage collection (computer science)",
    "Reference counting",
    "Tracing garbage collection",
    "Type inference",
    "Pattern matching (programming languages)",
    "Algebraic data type",
    "Generics in Java",
    "Template metaprogramming",
    "Metaprogramming",
    "Macro (computer science)",
    "Continuation-passing style",
    "Tail call",
    "LLVM",
    "Static single assignment form",
    "Data-flow analysis",
    "Loop optimization",
    "Inlining",
    "Escape analysis",

    # ── Functional programming ────────────────────────────────────────────────
    "Functional programming",
    "Pure function",
    "Immutable object",
    "Referential transparency",
    "Monad (functional programming)",
    "Functor (functional programming)",
    "Currying",
    "Partial application",
    "Lazy evaluation",
    "Higher-order function",
    "Closure (computer programming)",
    "Map (higher-order function)",
    "Filter (higher-order function)",
    "Fold (higher-order function)",
    "Function composition",
    "Persistent data structure",
    "Algebraic data type",
    "Pattern matching (programming languages)",
    "Hindley–Milner type system",

    # ── Distributed systems ───────────────────────────────────────────────────
    "Distributed computing",
    "Distributed consensus",
    "Paxos (computer science)",
    "Raft (algorithm)",
    "Two-phase commit protocol",
    "Three-phase commit protocol",
    "Vector clock",
    "Gossip protocol",
    "Consistent hashing",
    "Replica",
    "Leader election",
    "Distributed hash table",
    "MapReduce",
    "Eventual consistency",
    "CRDT",
    "Zookeeper (software)",
    "etcd",
    "Message queue",
    "Apache Kafka",
    "RabbitMQ",
    "Publish–subscribe pattern",
    "Remote procedure call",
    "Service discovery",
    "Health check (computing)",
    "Backpressure",
    "Idempotence",

    # ── Memory management ─────────────────────────────────────────────────────
    "Memory management",
    "Virtual memory",
    "Memory paging",
    "Cache (computing)",
    "CPU cache",
    "Cache hierarchy",
    "Cache replacement policies",
    "Memory pool",
    "Stack-based memory allocation",
    "Heap (programming)",
    "Resource acquisition is initialization",
    "Dangling pointer",
    "Memory leak",
    "Buffer overflow",
    "Valgrind",

    # ── Machine learning / AI ─────────────────────────────────────────────────
    "Machine learning",
    "Deep learning",
    "Neural network (machine learning)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Gradient descent",
    "Backpropagation",
    "Overfitting",
    "Regularization (mathematics)",
    "Cross-validation (statistics)",
    "Decision tree",
    "Random forest",
    "Support vector machine",
    "K-nearest neighbors algorithm",
    "Naive Bayes classifier",
    "Logistic regression",
    "Linear regression",
    "Reinforcement learning",
    "Natural language processing",
    "Word embedding",
    "Generative adversarial network",
    "Variational autoencoder",
    "Transfer learning",
    "Batch normalization",
    "Dropout (neural networks)",
    "Stochastic gradient descent",
    "Adam (optimization algorithm)",
    "Learning rate",
    "Hyperparameter optimization",
    "Feature engineering",
    "Principal component analysis",
    "Dimensionality reduction",
    "K-means clustering",
    "DBSCAN",
    "Autoencoder",
    "Encoder-decoder model",
    "Beam search",
    "Monte Carlo tree search",
    "Q-learning",
    "Policy gradient method",

    # ── Advanced data structures ──────────────────────────────────────────────
    "Persistent data structure",
    "Purely functional data structure",
    "Cache-oblivious algorithm",
    "External memory algorithm",
    "Amortized analysis",
    "Potential method",
    "Banker's algorithm",
    "Dancing Links",
    "X + Y sorting",
    "Fractional cascading",
    "Wavelet tree",
    "Succinct data structure",
    "Compressed suffix array",

    # ── More algorithms ───────────────────────────────────────────────────────
    "Greedy algorithm",
    "Divide-and-conquer algorithm",
    "Backtracking",
    "Branch and bound",
    "Approximation algorithm",
    "Randomized algorithm",
    "Las Vegas algorithm",
    "Monte Carlo algorithm",
    "Online algorithm",
    "Streaming algorithm",
    "Network flow",
    "Maximum matching",
    "Stable matching",
    "Hungarian algorithm",
    "Simulated annealing",
    "Genetic algorithm",
    "Ant colony optimization",
    "Particle swarm optimization",
    "Integer programming",
    "Linear programming",
    "Simplex algorithm",
    "Convex optimization",
    "Newton's method in optimization",
    "Gradient descent",
    "Stochastic gradient descent",
    "Coordinate descent",
    "Expectation–maximization algorithm",

    # ── Computer architecture ─────────────────────────────────────────────────
    "Computer architecture",
    "Instruction set architecture",
    "RISC",
    "CISC",
    "Pipeline (computing)",
    "Superscalar processor",
    "Out-of-order execution",
    "Branch predictor",
    "Speculative execution",
    "Memory hierarchy",
    "Cache coherence",
    "MESI protocol",
    "Non-uniform memory access",
    "Vector processor",
    "Graphics processing unit",
    "FPGA",
    "Memory-mapped I/O",
    "DMA",

    # ── Operating systems ─────────────────────────────────────────────────────
    "Operating system",
    "Process (computing)",
    "Scheduling (computing)",
    "Round-robin scheduling",
    "Completely Fair Scheduler",
    "Interrupt",
    "System call",
    "File system",
    "Virtual file system",
    "Inode",
    "File descriptor",
    "Pipe (Unix)",
    "Unix philosophy",
    "POSIX",
    "Memory-mapped file",
    "Copy-on-write",
    "Demand paging",
    "Thrashing (computer science)",
    "Journaling file system",

    # ── Networking (more) ─────────────────────────────────────────────────────
    "Network socket",
    "Berkeley sockets",
    "Epoll",
    "I/O multiplexing",
    "Non-blocking I/O",
    "Zero-copy",
    "Nagle's algorithm",
    "TCP congestion control",
    "QUIC",
    "HTTP cookie",
    "HTTP cache",
    "Cache-Control",
    "MIME",
    "URL",
    "URI",
    "Proxy server",
    "Nginx",
    "Apache HTTP Server",
    "HAProxy",
    "Traefik (software)",
    "Service discovery",
    "Consul (software)",
    "Envoy (software)",

    # ── Programming paradigms ─────────────────────────────────────────────────
    "Programming paradigm",
    "Declarative programming",
    "Imperative programming",
    "Object-oriented programming",
    "Functional programming",
    "Logic programming",
    "Reactive programming",
    "Dataflow programming",
    "Aspect-oriented programming",
    "Event-driven programming",
    "Concurrent computing",
    "Metaprogramming",
    "Generic programming",
    "Reflective programming",

    # ── Software quality ──────────────────────────────────────────────────────
    "Software quality",
    "ISO/IEC 25010",
    "FURPS",
    "Coupling (computer programming)",
    "Cohesion (computer science)",
    "Software entropy",
    "Broken windows theory (software)",
    "Boy Scout Rule",
    "Demeter's law",
    "Command–query separation",
    "Idempotence",
    "Software rot",
    "Legacy code",
    "Brownfield (software development)",
    "Greenfield project",

    # ── Databases (more) ─────────────────────────────────────────────────────
    "Apache HBase",
    "Neo4j",
    "InfluxDB",
    "Apache Flink",
    "Apache Beam",
    "dbt (data build tool)",
    "Materialized view",
    "Database view",
    "Denormalization",
    "Star schema",
    "Snowflake schema",
    "Data lake",
    "Lambda architecture",
    "Kappa architecture",
    "Change data capture",
    "Debezium",
    "Database trigger",
    "Stored procedure",
    "Window function (SQL)",
    "Common table expression",
    "Recursive query",
    "JSONB",

    # ── Security (more) ──────────────────────────────────────────────────────
    "Zero trust security model",
    "SAML",
    "Kerberos (protocol)",
    "LDAP",
    "Active Directory",
    "Role-based access control",
    "Attribute-based access control",
    "Security token",
    "Multi-factor authentication",
    "Single sign-on",
    "Federated identity",
    "Intrusion detection system",
    "Web application firewall",
    "DDoS",
    "Botnet",
    "Ransomware",
    "Phishing",
    "Social engineering (security)",
    "Red team",
    "Blue team",
    "Threat modeling",
    "STRIDE",
    "DAST",
    "SAST",
    "Software composition analysis",
    "Supply chain attack",
    "Dependency confusion",

    # ── Math / theory (more) ─────────────────────────────────────────────────
    "Graph theory",
    "Combinatorics",
    "Number theory",
    "Modular arithmetic",
    "Cryptographic protocol",
    "Zero-knowledge proof",
    "Boolean algebra",
    "Logic gate",
    "Propositional logic",
    "First-order logic",
    "Set theory",
    "Lattice (order)",
    "Abstract algebra",
    "Group (mathematics)",
    "Ring (mathematics)",
    "Field (mathematics)",
    "Vector space",
    "Matrix (mathematics)",
    "Eigenvalues and eigenvectors",
    "Singular value decomposition",
    "Fourier analysis",
    "Laplace transform",
    "Statistics",
    "Hypothesis testing",
    "Markov chain",
    "Hidden Markov model",
    "Kalman filter",
    "Game theory",
    "Nash equilibrium",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 500+ hand-crafted axioms
# ═══════════════════════════════════════════════════════════════════════════════

EXTRA_AXIOMS = [

    # ── Concurrency patterns ──────────────────────────────────────────────────
    ("Mutex (mutual exclusion lock) ensures that only one thread can access a critical section at a time, preventing data races.",
     "concurrency", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Protecting shared mutable state accessed from multiple threads",
      "tradeoffs": "Contention causes thread blocking; risk of deadlock if locks are acquired in inconsistent order",
      "keywords": ["mutex", "lock", "critical section", "thread safety", "deadlock"]}),

    ("Semaphore controls access to a shared resource pool; counting semaphores allow N concurrent holders.",
     "concurrency", 0.96, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Rate-limiting concurrent resource access (DB connections, file handles)",
      "tradeoffs": "More complex than mutex; forgetting to release causes resource starvation",
      "keywords": ["semaphore", "concurrency", "resource pool", "signaling"]}),

    ("The Actor model isolates state inside actors that communicate only via async message passing, eliminating shared-state bugs.",
     "concurrency", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Highly concurrent, distributed systems; Akka, Erlang, Elixir",
      "tradeoffs": "Message-passing overhead; mailbox backpressure; harder to debug than synchronous code",
      "keywords": ["actor model", "Akka", "Erlang", "message passing", "concurrency"]}),

    ("Communicating Sequential Processes (CSP): concurrent programs composed of independent processes that communicate via synchronous channels.",
     "concurrency", 0.90, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Go goroutines/channels; pipeline parallelism",
      "keywords": ["CSP", "channels", "goroutines", "Go", "concurrency"]}),

    ("Lock-free data structures use atomic operations (CAS) instead of locks to allow concurrent progress without blocking.",
     "concurrency", 0.91, KnowledgeType.ALGORITHM,
     {"when_to_use": "High-contention hot paths where lock overhead dominates",
      "tradeoffs": "ABA problem; implementation complexity; hard to verify correctness",
      "time_complexity": "O(1) amortized for stack/queue operations",
      "keywords": ["lock-free", "CAS", "compare-and-swap", "atomic", "non-blocking"]}),

    ("Thread pool pre-allocates worker threads to amortize thread creation cost over many tasks.",
     "concurrency", 0.95, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "I/O-bound or CPU-bound work dispatched from a hot path",
      "tradeoffs": "Fixed pool size requires tuning; blocking tasks can starve non-blocking ones",
      "keywords": ["thread pool", "executor", "worker", "task queue"]}),

    ("Work-stealing schedulers let idle threads steal tasks from busy threads' queues, improving CPU utilization.",
     "concurrency", 0.90, KnowledgeType.ALGORITHM,
     {"when_to_use": "Fine-grained parallel divide-and-conquer (fork/join, Rayon, Java ForkJoinPool)",
      "keywords": ["work stealing", "fork/join", "parallel", "scheduler", "Rayon"]}),

    ("Read-write lock allows concurrent readers but exclusive writer access, improving throughput for read-heavy workloads.",
     "concurrency", 0.94, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Data structures read frequently but rarely mutated (caches, config)",
      "tradeoffs": "Writer starvation possible; more complex than plain mutex",
      "keywords": ["read-write lock", "RWLock", "shared lock", "exclusive lock"]}),

    ("Avoid holding a lock while doing I/O, sleeping, or calling external code to minimise lock contention and deadlock risk.",
     "concurrency", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["lock contention", "deadlock", "critical section", "I/O"]}),

    ("Double-checked locking for lazy singleton initialisation is broken without memory barriers; use language-level guarantees (e.g. std::call_once, module-level init).",
     "concurrency", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["singleton", "double-checked locking", "memory barrier", "thread safety"]}),

    ("Prefer immutable data shared across threads over mutable data protected by locks; immutability eliminates an entire class of concurrency bugs.",
     "concurrency", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["immutability", "thread safety", "shared state", "functional"]}),

    # ── Database patterns ─────────────────────────────────────────────────────
    ("Connection pooling reuses database connections across requests, avoiding the high per-connection setup cost in RDBMS.",
     "databases", 0.96, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Any application with a relational database (use PgBouncer, HikariCP, psycopg pool)",
      "tradeoffs": "Pool exhaustion causes request queuing; connections may become stale",
      "keywords": ["connection pool", "database", "HikariCP", "PgBouncer", "psycopg"]}),

    ("The N+1 query problem: fetching N parent records then issuing N separate child queries; fix with JOIN or eager loading.",
     "databases", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Any ORM-based data access layer",
      "tradeoffs": "JOINs can produce large result sets; balance with pagination",
      "keywords": ["N+1 query", "ORM", "JOIN", "eager loading", "lazy loading", "SQLAlchemy"]}),

    ("Database indexes speed up reads but slow down writes; index columns used in WHERE, JOIN, and ORDER BY clauses.",
     "databases", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Columns with high cardinality queried frequently",
      "tradeoffs": "Each index increases storage and write overhead; unused indexes hurt more than help",
      "keywords": ["index", "query performance", "B-tree", "covering index", "explain"]}),

    ("Use EXPLAIN / EXPLAIN ANALYZE to inspect query plans and identify sequential scans on large tables.",
     "databases", 0.95, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Optimising slow queries; after schema changes",
      "keywords": ["EXPLAIN", "query plan", "seq scan", "index scan", "PostgreSQL"]}),

    ("Database sharding partitions data horizontally across nodes; consistent hashing minimises re-distribution on node changes.",
     "databases", 0.92, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "When data volume or write throughput exceeds single-node capacity",
      "tradeoffs": "Cross-shard queries are expensive; operational complexity increases",
      "keywords": ["sharding", "horizontal partitioning", "consistent hashing", "scalability"]}),

    ("Optimistic locking detects write conflicts at commit time (version counter / ETag) instead of blocking readers.",
     "databases", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Low-contention workloads where conflicts are rare",
      "tradeoffs": "Conflict resolution falls on the application; retry logic required",
      "keywords": ["optimistic locking", "version column", "ETag", "CAS", "concurrency"]}),

    ("Pessimistic locking (SELECT FOR UPDATE) holds a row lock for the duration of a transaction, preventing concurrent modification.",
     "databases", 0.92, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "High-contention rows where conflicts are frequent and costly to retry",
      "tradeoffs": "Reduces concurrency; risk of deadlock",
      "keywords": ["pessimistic locking", "SELECT FOR UPDATE", "row lock", "deadlock"]}),

    ("Database migrations should be backward-compatible with the running application (expand/contract pattern) to enable zero-downtime deploys.",
     "databases", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Continuous deployment environments",
      "tradeoffs": "Requires two-phase migration; temporary dual-write complexity",
      "keywords": ["database migration", "zero-downtime", "expand contract", "Flyway", "Alembic"]}),

    ("Read replicas offload analytical and reporting queries from the primary database, reducing write-path latency.",
     "databases", 0.91, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Replication lag means replicas may be slightly stale",
      "keywords": ["read replica", "replication", "CQRS", "read scaling"]}),

    ("Always use parameterised queries or prepared statements; never concatenate user input into SQL strings to prevent SQL injection.",
     "databases", 0.99, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Every database-backed application",
      "keywords": ["SQL injection", "parameterised query", "prepared statement", "security"]}),

    # ── API design ────────────────────────────────────────────────────────────
    ("REST best practice: use nouns for resource URLs (/users/42), HTTP verbs for actions, and return appropriate status codes.",
     "api_design", 0.95, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Designing public or internal REST APIs",
      "keywords": ["REST", "HTTP", "URL design", "status codes", "resource"]}),

    ("API versioning strategies: URL path (/v1/), Accept header, or query param; URL path is most visible and cache-friendly.",
     "api_design", 0.91, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "URL versioning breaks REST purity; header versioning is cleaner but harder to test",
      "keywords": ["API versioning", "REST", "backward compatibility"]}),

    ("Pagination of list endpoints is mandatory for any resource that can grow unboundedly; prefer cursor-based over offset-based for large datasets.",
     "api_design", 0.94, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Any endpoint returning a collection",
      "tradeoffs": "Cursor pagination prevents arbitrary page jumping",
      "keywords": ["pagination", "cursor", "offset", "API", "performance"]}),

    ("Rate limiting protects API servers from abuse and ensures fair usage; implement with a token bucket or sliding window algorithm.",
     "api_design", 0.94, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Public APIs and multi-tenant services",
      "keywords": ["rate limiting", "token bucket", "sliding window", "throttling", "429"]}),

    ("Idempotent APIs allow clients to safely retry requests without side effects; POST operations should use a client-supplied idempotency key.",
     "api_design", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Payment APIs, order creation, any mutation that must not be duplicated",
      "keywords": ["idempotency", "retry", "idempotency key", "at-least-once delivery"]}),

    ("HATEOAS (Hypermedia as the Engine of Application State): responses include links to valid next actions, making APIs self-discoverable.",
     "api_design", 0.82, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Adds response size; rarely implemented fully in practice",
      "keywords": ["HATEOAS", "REST", "hypermedia", "links"]}),

    ("Use HTTP 4xx codes for client errors (400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Too Many Requests) and 5xx for server errors.",
     "api_design", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["HTTP status codes", "error handling", "REST", "API"]}),

    ("API responses should include a consistent error schema (error code, message, details) so clients can handle errors programmatically.",
     "api_design", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["error schema", "API error handling", "error response"]}),

    ("Use OpenAPI (Swagger) to document REST APIs; machine-readable specs enable SDK generation, mock servers, and contract testing.",
     "api_design", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["OpenAPI", "Swagger", "API documentation", "contract testing"]}),

    ("gRPC uses HTTP/2 and Protocol Buffers for efficient, strongly-typed, bidirectional-streaming RPC between services.",
     "api_design", 0.92, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Internal microservice communication where performance and type safety matter",
      "tradeoffs": "Binary protocol harder to debug; browser support requires gRPC-web proxy",
      "keywords": ["gRPC", "protobuf", "HTTP/2", "RPC", "microservices"]}),

    ("GraphQL lets clients request exactly the fields they need, eliminating over-fetching and under-fetching.",
     "api_design", 0.91, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "APIs consumed by multiple clients with different data needs",
      "tradeoffs": "N+1 queries require DataLoader; harder to cache than REST",
      "keywords": ["GraphQL", "over-fetching", "under-fetching", "DataLoader"]}),

    # ── System design ─────────────────────────────────────────────────────────
    ("CAP theorem: a distributed system can guarantee at most two of Consistency, Availability, and Partition Tolerance simultaneously.",
     "system_design", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Designing distributed data stores; choosing between CP (Zookeeper) and AP (Cassandra) systems",
      "keywords": ["CAP theorem", "consistency", "availability", "partition tolerance"]}),

    ("Consistent hashing maps keys to nodes on a virtual ring, minimising key remapping when nodes are added or removed.",
     "system_design", 0.93, KnowledgeType.ALGORITHM,
     {"when_to_use": "Distributed caches (Memcached), DHTs, load balancers with sticky sessions",
      "tradeoffs": "Hot spots on the ring without virtual nodes",
      "keywords": ["consistent hashing", "ring", "distributed cache", "DHT", "sharding"]}),

    ("Horizontal sharding strategies: range-based (contiguous key ranges → simple but hot spots), hash-based (uniform distribution but no range queries), directory-based (explicit map → flexible but single point of failure).",
     "system_design", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["sharding", "range shard", "hash shard", "directory shard"]}),

    ("Cache-aside (lazy loading): application checks cache first, fetches from DB on miss, then populates cache.",
     "system_design", 0.95, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Read-heavy workloads with occasional writes",
      "tradeoffs": "Cache misses hit the database; stale data until TTL expires",
      "keywords": ["cache-aside", "lazy loading", "Redis", "Memcached", "cache"]}),

    ("Write-through cache updates the cache synchronously on every write, keeping cache and DB consistent at the cost of write latency.",
     "system_design", 0.92, KnowledgeType.DESIGN_PATTERN,
     {"tradeoffs": "Every write goes to both cache and DB; unnecessary caching of infrequently-read data",
      "keywords": ["write-through", "cache consistency", "write latency"]}),

    ("Write-behind (write-back) cache buffers writes in the cache and flushes to DB asynchronously, improving write throughput at the risk of data loss.",
     "system_design", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"tradeoffs": "Risk of data loss on cache failure before flush; complex consistency model",
      "keywords": ["write-behind", "write-back", "async write", "cache", "throughput"]}),

    ("Cache invalidation is one of the two hard problems in CS; prefer TTL-based expiry plus explicit invalidation on mutation.",
     "system_design", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["cache invalidation", "TTL", "stale data", "cache"]}),

    ("CDN (Content Delivery Network) caches static and dynamic content at edge nodes close to users, reducing origin server load and latency.",
     "system_design", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Static assets, API responses with Cache-Control headers, video streaming",
      "keywords": ["CDN", "edge cache", "latency", "Cloudflare", "Fastly"]}),

    ("Design for failure: assume every downstream call will eventually fail; implement retries, timeouts, and circuit breakers.",
     "system_design", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["resilience", "fault tolerance", "retry", "timeout", "circuit breaker"]}),

    ("Back-of-envelope estimation: 1 ms L1 cache miss, 100 μs RAM access, 0.5 ms SSD, 10 ms HDD, 150 ms cross-datacenter RTT.",
     "system_design", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["latency numbers", "estimation", "L1 cache", "SSD", "HDD", "system design interview"]}),

    ("The two-generals problem proves that reliable agreement over an unreliable channel is unsolvable; practical distributed consensus (Paxos, Raft) requires assumptions about partial synchrony.",
     "system_design", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["two generals", "consensus", "Paxos", "Raft", "distributed systems"]}),

    # ── Error handling ────────────────────────────────────────────────────────
    ("Fail fast: validate inputs and preconditions at the entry point and raise errors immediately rather than propagating invalid state.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["fail fast", "input validation", "defensive programming", "preconditions"]}),

    ("Circuit breaker pattern: after a threshold of failures, open the circuit and return fast errors instead of hammering an unavailable service.",
     "best_practices", 0.94, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Service-to-service calls in microservice architectures",
      "tradeoffs": "False positives open the circuit prematurely; half-open state requires careful probing",
      "keywords": ["circuit breaker", "resilience", "microservices", "Hystrix", "Resilience4j"]}),

    ("Retry with exponential backoff and jitter prevents thundering-herd storms when a dependency recovers.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Transient network errors, rate-limited APIs, database timeouts",
      "tradeoffs": "Long retry chains increase end-to-end latency; set a max retry limit",
      "keywords": ["exponential backoff", "jitter", "retry", "thundering herd"]}),

    ("Dead letter queue (DLQ): messages that fail processing after N retries are moved to a DLQ for inspection and reprocessing.",
     "best_practices", 0.92, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Message-queue consumers where some messages are persistently malformed",
      "keywords": ["dead letter queue", "DLQ", "SQS", "Kafka", "error handling"]}),

    ("Graceful degradation: when a non-critical dependency is unavailable, serve a degraded but functional response rather than a hard failure.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Recommendation engines, social features, non-critical enrichment",
      "keywords": ["graceful degradation", "fallback", "resilience", "availability"]}),

    ("Structured logging (JSON) enables machine-parseable log analysis; always include trace-id, request-id, and service name.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Any production service; required for distributed tracing correlation",
      "keywords": ["structured logging", "JSON logs", "trace-id", "observability"]}),

    ("Error messages exposed to clients should never reveal internal stack traces, database schemas, or system paths.",
     "best_practices", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["error message security", "information disclosure", "OWASP"]}),

    ("Use error codes alongside human-readable messages in API responses so clients can branch on error type without string parsing.",
     "best_practices", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["error codes", "API error handling", "client error handling"]}),

    # ── Memory management ─────────────────────────────────────────────────────
    ("RAII (Resource Acquisition Is Initialisation): bind resource lifetime to object scope; destructor/drop releases resources automatically.",
     "memory_management", 0.95, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "C++, Rust; Python context managers follow the same principle",
      "keywords": ["RAII", "destructor", "drop", "Rust", "C++", "resource management"]}),

    ("Reference counting enables deterministic deallocation but cannot collect cycles; use weak references to break cycles (Python weakref, Rust Weak).",
     "memory_management", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["reference counting", "cycle", "weak reference", "CPython", "ARC"]}),

    ("Tracing garbage collectors (mark-and-sweep, generational GC) collect cyclic garbage but introduce stop-the-world pauses.",
     "memory_management", 0.92, KnowledgeType.ALGORITHM,
     {"when_to_use": "Java, Go, C#, Python (for cycle collection supplement to refcount)",
      "tradeoffs": "GC pauses cause latency spikes; tune heap size and GC mode",
      "keywords": ["garbage collection", "mark-and-sweep", "generational GC", "stop-the-world"]}),

    ("Memory pool (arena allocator) pre-allocates a large slab and allocates objects from it; dramatically reduces allocation overhead and fragmentation.",
     "memory_management", 0.91, KnowledgeType.ALGORITHM,
     {"when_to_use": "High-frequency short-lived object allocation (parsers, game engines)",
      "keywords": ["memory pool", "arena allocator", "slab allocator", "fragmentation"]}),

    ("Stack allocation is orders of magnitude faster than heap allocation; prefer stack-allocated values for small, short-lived data.",
     "memory_management", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["stack allocation", "heap allocation", "performance", "cache locality"]}),

    ("Cache locality: accessing memory sequentially (arrays) is dramatically faster than random access (linked lists) due to CPU cache prefetching.",
     "memory_management", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["cache locality", "spatial locality", "CPU cache", "array", "AoS vs SoA"]}),

    ("False sharing: two threads writing to different fields in the same cache line cause constant cache invalidation; pad structs to cache-line boundaries.",
     "memory_management", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["false sharing", "cache line", "padding", "concurrent performance"]}),

    # ── Functional programming ────────────────────────────────────────────────
    ("Pure functions have no side effects and always return the same output for the same input; they are trivially testable, composable, and parallelisable.",
     "functional_programming", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["pure function", "side effect", "referential transparency", "functional"]}),

    ("Immutable data structures eliminate an entire class of bugs caused by shared mutable state; prefer immutability by default.",
     "functional_programming", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["immutability", "persistent data structure", "functional", "thread safety"]}),

    ("Referential transparency: an expression can be replaced by its value without changing program behaviour, enabling safe refactoring and memoisation.",
     "functional_programming", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["referential transparency", "pure function", "memoisation"]}),

    ("A monad is a design pattern that sequences computations with context (Maybe/Option for nullability, Result/Either for error handling, IO for effects).",
     "functional_programming", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Haskell, Scala, Rust Result/Option, Kotlin arrow-kt",
      "keywords": ["monad", "Maybe", "Result", "Either", "IO", "functional"]}),

    ("Currying transforms a function of multiple arguments into a sequence of single-argument functions, enabling partial application.",
     "functional_programming", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["currying", "partial application", "higher-order function", "functional"]}),

    ("Lazy evaluation defers computation until the result is needed, enabling infinite data structures and avoiding unnecessary work.",
     "functional_programming", 0.91, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Haskell (default), Python generators, Rust iterators",
      "keywords": ["lazy evaluation", "generator", "iterator", "infinite sequence"]}),

    ("Functor is any structure that can be mapped over; if a type implements fmap/map, it's a functor (List, Option, Result).",
     "functional_programming", 0.87, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["functor", "map", "fmap", "functional", "category theory"]}),

    ("Tail-call optimisation (TCO) converts tail-recursive calls into loops, preventing stack overflow in recursive algorithms.",
     "functional_programming", 0.91, KnowledgeType.ALGORITHM,
     {"when_to_use": "Scheme, Haskell, Kotlin (tailrec), Scala (tailrec); NOT Python",
      "keywords": ["tail call", "TCO", "recursion", "stack overflow"]}),

    ("Function composition: (f ∘ g)(x) = f(g(x)); building complex operations from simple, composable functions is the cornerstone of functional design.",
     "functional_programming", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["function composition", "pipeline", "compose", "pipe", "functional"]}),

    # ── Compiler / language internals ─────────────────────────────────────────
    ("JIT (Just-In-Time) compilation compiles hot code paths to native code at runtime; provides near-native performance while retaining dynamic language flexibility.",
     "compiler_internals", 0.93, KnowledgeType.ALGORITHM,
     {"when_to_use": "JVM (HotSpot), V8, LuaJIT, PyPy, .NET CLR",
      "tradeoffs": "JIT warmup time; increased memory usage for compiled code",
      "keywords": ["JIT", "just-in-time", "HotSpot", "V8", "PyPy", "native code"]}),

    ("Escape analysis determines whether an object's lifetime is bounded to the current stack frame; non-escaping objects can be stack-allocated.",
     "compiler_internals", 0.88, KnowledgeType.ALGORITHM,
     {"keywords": ["escape analysis", "stack allocation", "heap allocation", "JVM", "Go"]}),

    ("Type inference (Hindley-Milner): the compiler deduces types without explicit annotations, providing the safety of static typing with the brevity of dynamic typing.",
     "compiler_internals", 0.91, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Haskell, ML, Rust, Kotlin, Swift, TypeScript",
      "keywords": ["type inference", "Hindley-Milner", "static typing", "Rust", "Haskell"]}),

    ("Pattern matching on algebraic data types allows exhaustive, readable case analysis; compilers can warn on non-exhaustive patterns.",
     "compiler_internals", 0.92, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Rust (match), Haskell, Scala, Swift, Python 3.10+ (match/case)",
      "keywords": ["pattern matching", "ADT", "exhaustive match", "Rust", "Haskell"]}),

    ("Inlining small functions eliminates call overhead and enables further optimisations like constant folding across call boundaries.",
     "compiler_internals", 0.89, KnowledgeType.ALGORITHM,
     {"keywords": ["inlining", "function call overhead", "compiler optimisation", "LTO"]}),

    # ── Distributed systems ───────────────────────────────────────────────────
    ("Raft consensus: a leader is elected to manage log replication; leader appends entries and commits them once a majority acknowledges.",
     "distributed_systems", 0.93, KnowledgeType.ALGORITHM,
     {"when_to_use": "Distributed key-value stores, etcd, CockroachDB, TiKV",
      "keywords": ["Raft", "consensus", "leader election", "log replication", "etcd"]}),

    ("Paxos consensus: a two-phase protocol (Prepare/Promise, Accept/Accepted) for reaching agreement in the presence of failures.",
     "distributed_systems", 0.92, KnowledgeType.ALGORITHM,
     {"when_to_use": "Google Chubby, Apache Zookeeper, distributed databases",
      "keywords": ["Paxos", "consensus", "distributed", "Zookeeper"]}),

    ("Two-phase commit (2PC) provides atomic commit across distributed participants but blocks if the coordinator crashes after the prepare phase.",
     "distributed_systems", 0.91, KnowledgeType.ALGORITHM,
     {"tradeoffs": "Blocking on coordinator failure; use Saga pattern as alternative for long-running transactions",
      "keywords": ["2PC", "two-phase commit", "distributed transaction", "blocking"]}),

    ("Vector clocks track causality in distributed systems; if v(a) < v(b) then event a happened-before event b.",
     "distributed_systems", 0.90, KnowledgeType.ALGORITHM,
     {"keywords": ["vector clock", "happened-before", "causality", "Lamport clock"]}),

    ("Gossip protocol disseminates state updates across a cluster in O(log N) rounds, achieving eventual consistency without central coordination.",
     "distributed_systems", 0.89, KnowledgeType.ALGORITHM,
     {"when_to_use": "Membership, failure detection, anti-entropy (Cassandra, Riak)",
      "keywords": ["gossip protocol", "epidemic protocol", "Cassandra", "eventual consistency"]}),

    ("The Saga pattern decomposes a long-running transaction into a sequence of local transactions each with a compensating rollback transaction.",
     "distributed_systems", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Microservices where 2PC is impractical; e-commerce order processing",
      "tradeoffs": "Complex compensating logic; visibility of intermediate state",
      "keywords": ["Saga", "distributed transaction", "compensating transaction", "microservices"]}),

    ("CRDTs (Conflict-free Replicated Data Types) allow replicas to be updated concurrently without coordination; they converge by mathematical construction.",
     "distributed_systems", 0.88, KnowledgeType.ALGORITHM,
     {"when_to_use": "Collaborative editing, distributed counters, shopping carts (Riak, Redis CRDT)",
      "keywords": ["CRDT", "conflict-free", "eventual consistency", "distributed"]}),

    ("The outbox pattern: write business events to an outbox table in the same transaction as the domain mutation; a relay publishes them to the message broker.",
     "distributed_systems", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Guaranteeing at-least-once event publishing from a database-backed service",
      "keywords": ["outbox pattern", "transactional outbox", "at-least-once", "Debezium"]}),

    ("Idempotent consumers: each message should be processed at most once in effect even if delivered multiple times (deduplicate by message ID).",
     "distributed_systems", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["idempotency", "at-least-once", "message deduplication", "Kafka", "SQS"]}),

    ("Backpressure prevents a fast producer from overwhelming a slow consumer; implemented via bounded queues, reactive streams, or async semaphores.",
     "distributed_systems", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["backpressure", "flow control", "reactive streams", "async", "queue"]}),

    # ── Security / crypto ─────────────────────────────────────────────────────
    ("Hash passwords with a slow, salted adaptive KDF (bcrypt, scrypt, or Argon2id); never store plaintext or fast hashes like MD5/SHA-1.",
     "security", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["bcrypt", "Argon2", "password hashing", "KDF", "salt"]}),

    ("JWT (JSON Web Token) encodes claims in a signed (or encrypted) token; always verify the signature and check exp/iss/aud claims.",
     "security", 0.95, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Cannot be revoked without a blocklist; keep tokens short-lived",
      "keywords": ["JWT", "token", "authentication", "signature", "JOSE"]}),

    ("TLS 1.3 removes weak cipher suites and reduces handshake to 1-RTT; always enforce TLS 1.2+ minimum in production.",
     "security", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["TLS", "HTTPS", "cipher suite", "forward secrecy"]}),

    ("Content Security Policy (CSP) headers prevent XSS by whitelisting allowed script sources; combine with HttpOnly and Secure cookie flags.",
     "security", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["CSP", "XSS", "Content-Security-Policy", "HttpOnly", "cookie"]}),

    ("CSRF tokens must be unpredictable, tied to the session, and verified on every state-changing request.",
     "security", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["CSRF", "XSRF", "SameSite cookie", "token"]}),

    ("Principle of Least Privilege: processes and users should be granted only the minimum permissions required to perform their function.",
     "security", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["least privilege", "RBAC", "IAM", "security"]}),

    ("Secrets (API keys, DB passwords) must never be committed to version control; use environment variables, Vault, or a secrets manager.",
     "security", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["secrets management", "Vault", ".env", "environment variables"]}),

    ("Defence in depth: apply multiple independent security controls so that the failure of any single control does not compromise the system.",
     "security", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["defence in depth", "layered security", "security controls"]}),

    # ── Performance & profiling ───────────────────────────────────────────────
    ("Profile before optimising: use a profiler (cProfile, perf, async-profiler) to identify the actual bottleneck; intuition is often wrong.",
     "performance", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["profiling", "cProfile", "perf", "optimisation", "bottleneck"]}),

    ("Amdahl's Law: the maximum speedup from parallelising a fraction p of work is 1 / ((1-p) + p/N); even small serial portions cap total speedup.",
     "performance", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Amdahl's law", "parallelism", "speedup", "serial fraction"]}),

    ("The fastest code is code that doesn't run: eliminate unnecessary work before optimising the necessary work.",
     "performance", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["performance", "dead code elimination", "unnecessary work"]}),

    ("Vectorisation (SIMD) processes multiple data elements per CPU instruction; write tight loops over arrays for auto-vectorisation.",
     "performance", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["SIMD", "vectorisation", "AVX", "auto-vectorisation", "performance"]}),

    ("Database query performance: retrieving 1 million rows sequentially is far slower than retrieving 100 rows from a well-indexed query; always bound query result sets.",
     "performance", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["query performance", "index", "LIMIT", "pagination", "database"]}),

    # ── Testing ────────────────────────────────────────────────────────────────
    ("The test pyramid: many unit tests (fast, isolated), fewer integration tests, even fewer end-to-end tests; inverting the pyramid leads to slow, brittle CI.",
     "testing", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["test pyramid", "unit test", "integration test", "E2E", "CI"]}),

    ("Arrange-Act-Assert (AAA) structure makes tests readable: set up state, invoke the unit, assert the outcome.",
     "testing", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["AAA", "arrange act assert", "test structure", "unit testing"]}),

    ("Property-based testing (QuickCheck, Hypothesis) generates random inputs to find edge cases that hand-written examples miss.",
     "testing", 0.91, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Pure functions, data transformations, serialisation/deserialisation round-trips",
      "keywords": ["property-based testing", "QuickCheck", "Hypothesis", "fuzzing"]}),

    ("Test doubles: stubs return canned data, mocks verify interactions, spies record calls, fakes are lightweight implementations.",
     "testing", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["test double", "stub", "mock", "spy", "fake", "unittest.mock"]}),

    ("Mutation testing verifies test quality by introducing code mutations; tests that don't catch mutations are insufficiently discriminating.",
     "testing", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["mutation testing", "test quality", "mutant", "PIT", "mutmut"]}),

    ("Contract testing (Pact) ensures that API producers and consumers agree on the interface, catching breaking changes before integration.",
     "testing", 0.89, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["contract testing", "Pact", "consumer-driven contract", "API"]}),

    # ── DevOps / observability ─────────────────────────────────────────────────
    ("Distributed tracing (OpenTelemetry, Jaeger) propagates a trace-id through all service calls, enabling end-to-end latency analysis.",
     "devops", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["distributed tracing", "OpenTelemetry", "Jaeger", "trace-id", "observability"]}),

    ("The three pillars of observability are metrics, logs, and traces; a system is observable if you can understand its internal state from external outputs.",
     "devops", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["observability", "metrics", "logs", "traces", "SRE"]}),

    ("SLO (Service Level Objective) is a measurable reliability target (e.g., 99.9% availability); error budgets derived from SLOs gate risky deployments.",
     "devops", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["SLO", "SLA", "error budget", "reliability", "SRE"]}),

    ("Blue-green deployment keeps two identical environments; traffic switches to the new version atomically, enabling instant rollback.",
     "devops", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["blue-green deployment", "zero-downtime", "rollback", "deployment"]}),

    ("Canary release sends a small fraction of traffic to the new version; monitors metrics before gradually increasing the percentage.",
     "devops", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["canary release", "progressive delivery", "feature flag", "deployment"]}),

    ("Chaos engineering intentionally injects failures in production to verify system resilience and surface hidden weaknesses.",
     "devops", 0.89, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Netflix Chaos Monkey, AWS Fault Injection Simulator",
      "keywords": ["chaos engineering", "fault injection", "resilience", "Netflix"]}),

    ("Infrastructure as Code (IaC): manage servers and configuration with code (Terraform, Pulumi); enables reproducible, version-controlled infrastructure.",
     "devops", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["IaC", "Terraform", "Pulumi", "infrastructure as code", "reproducible"]}),

    ("Container images should be minimal and pinned to a specific digest, not 'latest', to ensure reproducible deployments.",
     "devops", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Docker", "container image", "immutable image", "digest", "Dockerfile"]}),

    # ── Algorithms (additional) ───────────────────────────────────────────────
    ("Timsort (Python's default sort): hybrid merge/insertion sort that exploits existing runs in real-world data for O(n) best case.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n log n) worst, O(n) best",
      "space_complexity": "O(n)",
      "keywords": ["Timsort", "Python sort", "Java sort", "adaptive sort"]}),

    ("Counting sort runs in O(n + k) for integer keys bounded by k; faster than comparison sorts when k is small.",
     "algorithms", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n + k)",
      "space_complexity": "O(k)",
      "when_to_use": "Sorting small-range integers (e.g., frequency counts, age buckets)",
      "keywords": ["counting sort", "linear sort", "integer sort"]}),

    ("Radix sort processes digits from least to most significant using a stable subroutine (counting sort), achieving O(nk) for k-digit numbers.",
     "algorithms", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(nk) where k = number of digits",
      "space_complexity": "O(n + k)",
      "keywords": ["radix sort", "LSD radix sort", "MSD radix sort", "linear sort"]}),

    ("Dijkstra's algorithm finds shortest paths from a source in O((V+E) log V) using a min-heap; requires non-negative edge weights.",
     "algorithms", 0.98, KnowledgeType.ALGORITHM,
     {"time_complexity": "O((V+E) log V) with binary heap",
      "space_complexity": "O(V)",
      "when_to_use": "Road networks, routing protocols, game pathfinding",
      "tradeoffs": "Incorrect with negative weights; use Bellman-Ford instead",
      "keywords": ["Dijkstra", "shortest path", "graph", "heap", "non-negative weights"]}),

    ("A* search adds a heuristic h(n) to Dijkstra to guide exploration toward the goal; optimal when h is admissible (never overestimates).",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(E) best case with perfect heuristic",
      "when_to_use": "Grid pathfinding, game AI, navigation systems",
      "tradeoffs": "Memory-intensive; suboptimal heuristic yields suboptimal path",
      "keywords": ["A*", "heuristic", "admissible", "pathfinding", "game AI"]}),

    ("Bellman-Ford computes shortest paths even with negative edge weights in O(VE); also detects negative-weight cycles.",
     "algorithms", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(VE)",
      "space_complexity": "O(V)",
      "when_to_use": "Graphs with negative edges; currency arbitrage detection",
      "keywords": ["Bellman-Ford", "negative weights", "shortest path", "negative cycle"]}),

    ("Floyd-Warshall computes all-pairs shortest paths in O(V³) via dynamic programming on intermediate vertices.",
     "algorithms", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(V³)",
      "space_complexity": "O(V²)",
      "when_to_use": "Dense graphs where all-pairs distances are needed; transitive closure",
      "keywords": ["Floyd-Warshall", "all-pairs shortest path", "dynamic programming"]}),

    ("KMP (Knuth-Morris-Pratt) string search skips redundant comparisons using a failure function, achieving O(n+m) overall.",
     "algorithms", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n + m) where n = text length, m = pattern length",
      "space_complexity": "O(m)",
      "keywords": ["KMP", "string search", "failure function", "pattern matching"]}),

    ("Suffix arrays allow O(m log n) substring search after O(n log n) construction; more cache-friendly than suffix trees.",
     "algorithms", 0.91, KnowledgeType.ALGORITHM,
     {"time_complexity": "Build O(n log n), search O(m log n)",
      "space_complexity": "O(n)",
      "keywords": ["suffix array", "suffix tree", "substring search", "LCP array"]}),

    ("Bloom filter: probabilistic set membership test using k hash functions; no false negatives, tunable false positive rate, O(1) lookup.",
     "algorithms", 0.93, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(k) insert and lookup",
      "space_complexity": "O(m) bits",
      "when_to_use": "Cache negative lookups, network duplicate detection, spell checkers",
      "tradeoffs": "Cannot delete elements; returns false positives at a tunable rate",
      "keywords": ["Bloom filter", "probabilistic", "false positive", "set membership"]}),

    ("Union-Find (Disjoint Set Union) supports near-O(1) amortised union and find with path compression and union by rank.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(α(n)) amortised per operation (inverse Ackermann)",
      "space_complexity": "O(n)",
      "when_to_use": "Connected components, Kruskal's MST, cycle detection",
      "keywords": ["union-find", "DSU", "disjoint set", "Kruskal", "connected components"]}),

    ("Segment tree supports range queries and point/range updates in O(log n); use lazy propagation for range updates.",
     "data_structures", 0.94, KnowledgeType.ALGORITHM,
     {"time_complexity": "Build O(n), query/update O(log n)",
      "space_complexity": "O(n)",
      "when_to_use": "Range sum, range min/max, range GCD, interval scheduling",
      "keywords": ["segment tree", "range query", "lazy propagation"]}),

    ("Fenwick tree (Binary Indexed Tree) supports prefix-sum queries and point updates in O(log n) with a simpler implementation than segment tree.",
     "data_structures", 0.93, KnowledgeType.ALGORITHM,
     {"time_complexity": "Build O(n log n), query/update O(log n)",
      "space_complexity": "O(n)",
      "when_to_use": "Prefix sums, frequency tables, order statistics",
      "keywords": ["Fenwick tree", "BIT", "binary indexed tree", "prefix sum"]}),

    ("LRU cache evicts the least recently used entry on overflow; implement with a HashMap + doubly-linked list for O(1) get and put.",
     "data_structures", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(1) get and put",
      "space_complexity": "O(capacity)",
      "when_to_use": "Page replacement, DNS caches, memoisation with bounded memory",
      "code_snippet": """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)""",
      "language": "Python",
      "keywords": ["LRU", "cache", "OrderedDict", "eviction"]}),

    # ── OOP & design principles (more depth) ──────────────────────────────────
    ("Tell, Don't Ask: instead of querying an object's state and then acting, tell the object what to do; keeps behaviour co-located with data.",
     "best_practices", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["tell don't ask", "encapsulation", "OOP", "Law of Demeter"]}),

    ("Favour small, focused interfaces over large ones (Interface Segregation Principle); clients should not depend on methods they don't use.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["ISP", "SOLID", "interface", "cohesion"]}),

    ("Prefer returning errors as values (Result types) over throwing exceptions for expected failure modes; reserve exceptions for truly exceptional conditions.",
     "best_practices", 0.91, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Rust (Result), Go (error return), Kotlin (Either), Python (raise only for unexpected)",
      "keywords": ["Result type", "error handling", "exceptions", "Rust", "Go"]}),

    ("Avoid boolean method parameters that flip behaviour; use two distinct methods or an enum to make call sites self-documenting.",
     "best_practices", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["boolean parameter", "code smell", "readability", "API design"]}),

    ("The Null Object pattern replaces null checks by returning a do-nothing object that conforms to the expected interface.",
     "best_practices", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["Null Object", "design pattern", "null check", "NullPointerException"]}),

    # ── Python-specific ───────────────────────────────────────────────────────
    ("Python generators use 'yield' to produce values lazily, enabling O(1) memory iteration over large or infinite sequences.",
     "python", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Reading large files, streaming pipelines, infinite sequences",
      "code_snippet": "def fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b",
      "language": "Python",
      "keywords": ["generator", "yield", "lazy evaluation", "memory efficiency", "Python"]}),

    ("Python's GIL (Global Interpreter Lock) prevents true CPU parallelism in threads; use multiprocessing or asyncio for CPU-bound or I/O-bound concurrency respectively.",
     "python", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["GIL", "multiprocessing", "asyncio", "Python", "concurrency"]}),

    ("Use dataclasses or attrs for data-holding classes to auto-generate __init__, __repr__, and __eq__ without boilerplate.",
     "python", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["dataclass", "attrs", "Python", "boilerplate"]}),

    ("Context managers (with statement, __enter__/__exit__) guarantee resource cleanup even on exceptions; use contextlib.contextmanager for generator-based CMs.",
     "python", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["context manager", "with", "__enter__", "__exit__", "RAII", "Python"]}),

    ("List comprehensions and generator expressions are more Pythonic and typically faster than equivalent for-loop constructions.",
     "python", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["list comprehension", "generator expression", "Pythonic", "performance"]}),

    ("Use typing module annotations and mypy/pyright for static type checking; types serve as machine-verified documentation.",
     "python", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["type hints", "mypy", "pyright", "static typing", "Python"]}),

    # ── Go-specific ───────────────────────────────────────────────────────────
    ("Go goroutines are lightweight green threads multiplexed on OS threads; spawn millions without exhausting memory.",
     "golang", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["goroutine", "Go", "concurrency", "green thread", "M:N scheduling"]}),

    ("Go channels are the primary synchronisation primitive; 'do not communicate by sharing memory, share memory by communicating'.",
     "golang", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["channel", "Go", "CSP", "goroutine", "concurrency"]}),

    ("Go's defer runs cleanup code when the surrounding function returns, regardless of whether it returns normally or via panic.",
     "golang", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["defer", "Go", "cleanup", "panic", "RAII"]}),

    # ── Rust-specific ─────────────────────────────────────────────────────────
    ("Rust's ownership system enforces at compile time that every value has exactly one owner, eliminating use-after-free and double-free bugs.",
     "rust", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["ownership", "Rust", "memory safety", "borrow checker"]}),

    ("Rust lifetimes are compile-time annotations that ensure references never outlive the data they point to.",
     "rust", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["lifetime", "Rust", "borrow checker", "reference"]}),

    ("Rust's Result<T,E> and Option<T> types make error and null handling explicit and force callers to handle all cases.",
     "rust", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Result", "Option", "Rust", "error handling", "null safety"]}),

    # ── JavaScript / TypeScript ───────────────────────────────────────────────
    ("Prefer async/await over raw Promise chains for readable, sequential-looking asynchronous code in JavaScript.",
     "javascript", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["async/await", "Promise", "JavaScript", "asynchronous"]}),

    ("TypeScript's strict mode enables strict null checks, strict function types, and no implicit any; always enable it.",
     "javascript", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["TypeScript", "strict mode", "null safety", "type safety"]}),

    ("Event delegation attaches a single listener on a parent element instead of N listeners on N children, reducing memory usage.",
     "javascript", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["event delegation", "DOM", "JavaScript", "performance"]}),

    # ── Java / JVM ────────────────────────────────────────────────────────────
    ("Use the Diamond problem-safe approach in Java: prefer composition and interfaces over multiple inheritance.",
     "java", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Java", "composition", "interface", "multiple inheritance"]}),

    ("Java streams (java.util.stream) enable declarative, lazy data processing pipelines; use parallel() only when benchmarks confirm speedup.",
     "java", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Java streams", "lazy evaluation", "parallel stream", "functional"]}),

    ("Prefer try-with-resources (AutoCloseable) over finally blocks for resource cleanup in Java.",
     "java", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["try-with-resources", "AutoCloseable", "Java", "RAII"]}),

    # ── Cloud-native ──────────────────────────────────────────────────────────
    ("Twelve-factor app principles: one codebase, explicit dependencies, config in environment, stateless processes, disposability, dev/prod parity.",
     "cloud", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["twelve-factor", "12-factor", "cloud-native", "stateless"]}),

    ("Kubernetes pods are ephemeral; never store state on a pod's local filesystem — use PersistentVolumes, databases, or object storage.",
     "cloud", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Kubernetes", "pod", "ephemeral", "persistent volume", "stateless"]}),

    ("Resource requests and limits in Kubernetes ensure fair CPU/memory allocation and prevent noisy-neighbour problems.",
     "cloud", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Kubernetes", "resource requests", "limits", "QoS"]}),

    ("Use readiness probes to gate traffic to a pod and liveness probes to restart crashed pods in Kubernetes.",
     "cloud", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Kubernetes", "readiness probe", "liveness probe", "health check"]}),

    ("Serverless functions (Lambda, Cloud Functions) are ideal for event-driven, stateless, infrequently-invoked workloads; cold starts are a concern for latency-sensitive paths.",
     "cloud", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["serverless", "Lambda", "cold start", "FaaS", "event-driven"]}),

    # ── Architecture patterns (more) ──────────────────────────────────────────
    ("Event sourcing stores state as an immutable sequence of domain events; current state is derived by replaying the event log.",
     "architecture", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Audit trails, time-travel debugging, CQRS projections",
      "tradeoffs": "Event schema evolution is complex; eventually consistent read models",
      "keywords": ["event sourcing", "event log", "CQRS", "audit trail"]}),

    ("CQRS separates the command (write) model from the query (read) model, allowing each to be optimised independently.",
     "architecture", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "High read-to-write ratio differences; complex domain models",
      "tradeoffs": "Additional complexity; eventual consistency between models",
      "keywords": ["CQRS", "command", "query", "read model", "write model"]}),

    ("Strangler fig pattern: incrementally migrate a monolith to microservices by routing traffic to the new service for specific features.",
     "architecture", 0.89, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["strangler fig", "migration", "monolith", "microservices", "incremental"]}),

    ("Anti-corruption layer translates between two domain models, protecting a clean domain from a legacy or external system's model.",
     "architecture", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["anti-corruption layer", "DDD", "bounded context", "translation"]}),

    ("Bulkhead pattern isolates components into pools so that failure in one pool does not cascade to others.",
     "architecture", 0.90, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Multi-tenant services; isolating slow third-party calls",
      "keywords": ["bulkhead", "isolation", "failure", "thread pool", "resilience"]}),

    ("Sidecar pattern deploys a helper container alongside the main application container in the same pod to handle cross-cutting concerns.",
     "architecture", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Envoy proxies, log shippers, certificate rotators in Kubernetes",
      "keywords": ["sidecar", "Kubernetes", "pod", "service mesh", "Istio"]}),

    ("API composition pattern: an aggregator service calls multiple downstream services and combines results, avoiding client-side scatter-gather.",
     "architecture", 0.87, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["API composition", "aggregator", "BFF", "Backend for Frontend"]}),

    ("Backend for Frontend (BFF) pattern: each client type (web, mobile) has its own backend API tailored to its specific needs.",
     "architecture", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["BFF", "Backend for Frontend", "API", "mobile", "web"]}),

    # ── Data streaming / pipelines ─────────────────────────────────────────────
    ("Apache Kafka partitions topics across brokers; each partition is an ordered, immutable log consumed independently by consumer groups.",
     "databases", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "High-throughput event streaming, log aggregation, change data capture",
      "tradeoffs": "Ordering guaranteed only within a partition; at-least-once delivery by default",
      "keywords": ["Kafka", "partition", "consumer group", "event streaming", "log"]}),

    ("Stream processing (Flink, Spark Streaming, Kafka Streams) enables real-time transformations on unbounded data streams.",
     "databases", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["stream processing", "Flink", "Kafka Streams", "real-time", "windowing"]}),

    ("Windowing in stream processing: tumbling windows (non-overlapping), sliding windows (overlapping), session windows (activity-based).",
     "databases", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["windowing", "tumbling window", "sliding window", "session window", "stream"]}),

    ("Exactly-once semantics in distributed messaging requires idempotent producers and transactional consumers; at-least-once is simpler and more common.",
     "databases", 0.89, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["exactly-once", "at-least-once", "idempotent", "Kafka transactions"]}),

    # ── Microservices (more) ──────────────────────────────────────────────────
    ("Service mesh (Istio, Linkerd) offloads cross-cutting concerns (mTLS, retries, tracing, load balancing) from application code to the infrastructure layer.",
     "architecture", 0.89, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["service mesh", "Istio", "Linkerd", "mTLS", "sidecar"]}),

    ("Health checks (liveness and readiness probes) allow orchestrators to route traffic away from unhealthy instances and restart stuck processes.",
     "architecture", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["health check", "liveness probe", "readiness probe", "Kubernetes"]}),

    ("Microservice boundaries should align with bounded contexts from Domain-Driven Design; team ownership maps one-to-one with services.",
     "architecture", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["microservices", "bounded context", "DDD", "Conway's law"]}),

    ("Conway's law: organisations design systems that mirror their communication structure; team topology drives architecture.",
     "architecture", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Conway's law", "team topology", "architecture", "organisation"]}),

    # ── SQL / query patterns ──────────────────────────────────────────────────
    ("Use covering indexes (index includes all projected columns) to avoid heap lookups for frequent queries.",
     "databases", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["covering index", "index-only scan", "PostgreSQL", "performance"]}),

    ("Partial indexes index only a subset of rows matching a predicate; smaller and faster than full indexes for skewed queries.",
     "databases", 0.89, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["partial index", "filtered index", "PostgreSQL", "index size"]}),

    ("CTEs (Common Table Expressions) with WITH improve readability of complex queries; materialised CTEs in PostgreSQL can affect the query plan.",
     "databases", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["CTE", "WITH clause", "SQL", "readability", "materialized"]}),

    ("Window functions (ROW_NUMBER, RANK, LAG, LEAD, SUM OVER) perform calculations across a partition without collapsing rows.",
     "databases", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["window function", "ROW_NUMBER", "RANK", "OVER", "PARTITION BY", "SQL"]}),

    ("Avoid SELECT *; explicitly list columns to prevent schema-change breakage, reduce data transfer, and enable covering index use.",
     "databases", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["SELECT *", "explicit columns", "SQL best practice"]}),

    ("Transactions should be as short as possible; long-running transactions hold locks and increase blocking and deadlock probability.",
     "databases", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["transaction", "short transaction", "lock", "deadlock", "RDBMS"]}),

    # ── Python advanced ───────────────────────────────────────────────────────
    ("Python's __slots__ declaration prevents per-instance __dict__ creation, reducing memory usage for classes with many instances.",
     "python", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["__slots__", "Python", "memory", "performance"]}),

    ("Use functools.lru_cache or functools.cache to memoize expensive pure functions with hashable arguments.",
     "python", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["lru_cache", "memoization", "functools", "Python", "cache"]}),

    ("Python's asyncio event loop runs coroutines concurrently on a single thread; await yields control at I/O boundaries.",
     "python", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["asyncio", "event loop", "coroutine", "await", "I/O bound", "Python"]}),

    ("Use pathlib.Path instead of os.path for file system operations; it is more readable and cross-platform.",
     "python", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["pathlib", "Path", "os.path", "Python", "file system"]}),

    ("Python's collections.defaultdict, Counter, and deque solve common patterns with optimised C implementations.",
     "python", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["defaultdict", "Counter", "deque", "collections", "Python"]}),

    ("Use Enum classes instead of string/integer constants to get type safety, iteration, and IDE autocomplete.",
     "python", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Enum", "Python", "constants", "type safety"]}),

    ("Profile Python code with cProfile + snakeviz or py-spy before optimising; rewriting in C extensions or using NumPy is usually faster than optimising pure Python.",
     "python", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["cProfile", "py-spy", "NumPy", "Cython", "profiling", "Python"]}),

    ("Virtual environments (venv, Poetry, uv) isolate project dependencies, preventing version conflicts between projects.",
     "python", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["venv", "Poetry", "uv", "virtual environment", "Python", "dependency"]}),

    # ── JavaScript / Node.js advanced ─────────────────────────────────────────
    ("The JavaScript event loop processes the call stack, then microtasks (Promises), then macrotasks (setTimeout, I/O) in each tick.",
     "javascript", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["event loop", "microtask", "macrotask", "Promise", "setTimeout", "JavaScript"]}),

    ("Node.js is single-threaded; CPU-bound tasks block the event loop — offload to worker threads or a separate process.",
     "javascript", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Node.js", "event loop", "worker threads", "CPU bound", "blocking"]}),

    ("Use WeakMap and WeakSet in JavaScript for metadata on objects without preventing garbage collection.",
     "javascript", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["WeakMap", "WeakSet", "JavaScript", "garbage collection", "memory"]}),

    ("TypeScript discriminated unions model tagged variants safely; narrow with a literal 'kind' or 'type' field.",
     "javascript", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["discriminated union", "TypeScript", "tagged union", "narrowing"]}),

    # ── Java advanced ─────────────────────────────────────────────────────────
    ("Java's CompletableFuture enables non-blocking async pipelines with thenApply, thenCompose, and exceptionally.",
     "java", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["CompletableFuture", "Java", "async", "non-blocking", "pipeline"]}),

    ("Java records (Java 16+) are immutable data carriers with auto-generated equals, hashCode, toString, and accessor methods.",
     "java", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["record", "Java", "immutable", "data carrier"]}),

    ("Java virtual threads (Project Loom, Java 21) are lightweight threads scheduled by the JVM, enabling millions of concurrent I/O operations.",
     "java", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["virtual thread", "Project Loom", "Java 21", "lightweight thread"]}),

    ("Always close JDBC ResultSets, Statements, and Connections in finally blocks or use try-with-resources to prevent connection leaks.",
     "java", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["JDBC", "connection leak", "try-with-resources", "ResultSet", "Java"]}),

    # ── Kotlin / JVM ─────────────────────────────────────────────────────────
    ("Kotlin coroutines are lightweight and cooperative; suspend functions pause at suspension points without blocking the OS thread.",
     "kotlin", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Kotlin", "coroutine", "suspend", "async", "non-blocking"]}),

    ("Kotlin's sealed classes model closed hierarchies enabling exhaustive when expressions without else branches.",
     "kotlin", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["sealed class", "Kotlin", "when expression", "exhaustive", "ADT"]}),

    ("Kotlin extension functions add methods to existing classes without inheritance; ideal for adding utility to third-party types.",
     "kotlin", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["extension function", "Kotlin", "utility", "third-party"]}),

    # ── Go advanced ───────────────────────────────────────────────────────────
    ("In Go, always check error return values; ignoring errors silently is the Go equivalent of swallowing exceptions.",
     "golang", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Go", "error handling", "error return", "nil error"]}),

    ("Go's context.Context propagates deadlines, cancellation signals, and request-scoped values across API boundaries.",
     "golang", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["context", "Go", "cancellation", "deadline", "timeout"]}),

    ("Use sync.Pool to amortize allocation cost for short-lived objects in hot paths in Go.",
     "golang", 0.89, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["sync.Pool", "Go", "allocation", "performance", "hot path"]}),

    ("Go interfaces are satisfied implicitly; a type implements an interface simply by having the required methods.",
     "golang", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["interface", "Go", "duck typing", "implicit"]}),

    # ── Rust advanced ─────────────────────────────────────────────────────────
    ("Rust's trait system enables zero-cost abstractions; generic code is monomorphised at compile time with no runtime dispatch overhead.",
     "rust", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["trait", "Rust", "monomorphisation", "zero-cost abstraction", "generics"]}),

    ("Rust's async/await builds on Futures; unlike goroutines, Rust async is zero-cost and requires an explicit runtime (Tokio, async-std).",
     "rust", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["async", "await", "Rust", "Tokio", "Future", "zero-cost"]}),

    ("Rust's Send and Sync traits mark types safe for inter-thread transfer and shared reference across threads; the compiler enforces these bounds.",
     "rust", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Send", "Sync", "Rust", "thread safety", "borrow checker"]}),

    # ── Observability / SRE ───────────────────────────────────────────────────
    ("Golden signals: latency, traffic, errors, and saturation are the four signals that matter most for service health monitoring.",
     "devops", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["golden signals", "latency", "traffic", "errors", "saturation", "SRE"]}),

    ("P99 latency is more important than mean latency for user-facing services; tail latencies govern the worst-case user experience.",
     "devops", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["P99", "tail latency", "percentile", "SLO", "user experience"]}),

    ("Alerting on symptoms (high error rate, high latency) is more actionable than alerting on causes (CPU spike, memory usage).",
     "devops", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["alerting", "symptoms vs causes", "SRE", "observability"]}),

    ("Runbooks document the standard response procedure for each alert, reducing mean time to resolve (MTTR) incidents.",
     "devops", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["runbook", "MTTR", "incident", "on-call", "SRE"]}),

    ("Post-mortems (blameless) after incidents focus on systemic causes, not individual blame, to drive lasting improvements.",
     "devops", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["post-mortem", "blameless", "incident", "root cause", "SRE"]}),

    # ── Testing advanced ──────────────────────────────────────────────────────
    ("Load testing (k6, Locust, JMeter) measures system behaviour under expected and peak traffic before production deployment.",
     "testing", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["load testing", "k6", "Locust", "JMeter", "performance test"]}),

    ("Snapshot testing captures component output and fails on unexpected changes; useful for UI and serialisation format regression.",
     "testing", 0.87, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["snapshot testing", "Jest", "regression", "UI testing"]}),

    ("Test isolation: each test should set up its own state and not depend on the order of test execution.",
     "testing", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["test isolation", "test order", "idempotent test", "unit testing"]}),

    ("Use test fixtures and factories (factory_boy, FactoryBot) to create test data consistently without duplicating setup code.",
     "testing", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["test fixture", "factory", "factory_boy", "FactoryBot", "test data"]}),

    # ── Build systems / tooling ───────────────────────────────────────────────
    ("Hermetic builds produce the same output regardless of the host environment; pin all dependency versions and use content-addressed caches.",
     "best_practices", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["hermetic build", "reproducible build", "Bazel", "Nix", "determinism"]}),

    ("Semantic versioning (MAJOR.MINOR.PATCH): increment MAJOR for breaking changes, MINOR for new features, PATCH for bug fixes.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["semver", "semantic versioning", "breaking change", "backward compatibility"]}),

    ("Pin exact dependency versions in lock files (poetry.lock, Cargo.lock, package-lock.json) for reproducible builds.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["lock file", "pin dependencies", "reproducible", "poetry.lock", "Cargo.lock"]}),

    ("Dependency confusion attack exploits package registries by publishing a malicious public package with the same name as a private one; use namespace scoping to prevent it.",
     "security", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["dependency confusion", "supply chain attack", "package registry", "namespace"]}),

    # ── Algorithms code snippets ───────────────────────────────────────────────
    ("Quicksort with Lomuto partition: choose last element as pivot, partition in-place; average O(n log n), worst O(n²) on sorted input.",
     "algorithms", 0.97, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n log n) average, O(n²) worst",
      "space_complexity": "O(log n) stack",
      "code_snippet": """def quicksort(arr, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        p = partition(arr, lo, hi)
        quicksort(arr, lo, p - 1)
        quicksort(arr, p + 1, hi)

def partition(arr, lo, hi):
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[hi] = arr[hi], arr[i+1]
    return i + 1""",
      "language": "Python",
      "keywords": ["quicksort", "Lomuto partition", "in-place", "pivot"]}),

    ("Heap sort: build max-heap in O(n) then extract elements in O(n log n); in-place, not stable.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n log n) all cases",
      "space_complexity": "O(1)",
      "code_snippet": """import heapq

def heapsort(arr):
    # Python heapq is min-heap; negate for max-heap sort
    h = [-x for x in arr]
    heapq.heapify(h)
    return [-heapq.heappop(h) for _ in range(len(h))]""",
      "language": "Python",
      "keywords": ["heapsort", "max-heap", "in-place", "O(n log n)"]}),

    ("Topological sort with Kahn's algorithm: repeatedly remove nodes with in-degree 0; detects cycles when queue empties with remaining nodes.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(V + E)",
      "space_complexity": "O(V)",
      "code_snippet": """from collections import deque

def topological_sort(graph, num_nodes):
    in_degree = [0] * num_nodes
    for u in range(num_nodes):
        for v in graph[u]:
            in_degree[v] += 1
    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return order if len(order) == num_nodes else []  # empty = cycle""",
      "language": "Python",
      "keywords": ["topological sort", "Kahn's algorithm", "DAG", "cycle detection"]}),

    ("Dijkstra's algorithm with a min-heap in Python: use heapq to extract the minimum-distance node efficiently.",
     "algorithms", 0.97, KnowledgeType.ALGORITHM,
     {"time_complexity": "O((V+E) log V)",
      "code_snippet": """import heapq

def dijkstra(graph, src, n):
    dist = [float('inf')] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist""",
      "language": "Python",
      "keywords": ["Dijkstra", "shortest path", "heapq", "graph"]}),

    ("BFS shortest path in an unweighted graph returns the minimum number of edges between source and target.",
     "algorithms", 0.97, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(V + E)",
      "code_snippet": """from collections import deque

def bfs_shortest_path(graph, src, dst):
    queue = deque([(src, [src])])
    visited = {src}
    while queue:
        node, path = queue.popleft()
        if node == dst:
            return path
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return []""",
      "language": "Python",
      "keywords": ["BFS", "shortest path", "unweighted", "graph"]}),

    ("Two-pointer technique solves many array problems in O(n) that naively require O(n²); pointers start at opposite ends or move together.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n)",
      "code_snippet": """def two_sum_sorted(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        s = arr[lo] + arr[hi]
        if s == target:
            return (lo, hi)
        elif s < target:
            lo += 1
        else:
            hi -= 1
    return None""",
      "language": "Python",
      "keywords": ["two pointer", "sliding window", "sorted array", "O(n)"]}),

    ("Sliding window technique maintains a contiguous sub-array or sub-string of variable or fixed length in O(n).",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n)",
      "code_snippet": """def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum""",
      "language": "Python",
      "keywords": ["sliding window", "subarray", "O(n)", "fixed window"]}),

    ("Backtracking explores all candidates recursively and prunes invalid branches early; template: choose, explore, unchoose.",
     "algorithms", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "Exponential in the worst case",
      "code_snippet": """def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i, n in enumerate(remaining):
            path.append(n)
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result""",
      "language": "Python",
      "keywords": ["backtracking", "permutations", "combinatorics", "recursion"]}),

    # ── Data structure code snippets ───────────────────────────────────────────
    ("Min-stack supports O(1) push, pop, and getMin by maintaining an auxiliary stack of current minimums.",
     "data_structures", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(1) for all operations",
      "space_complexity": "O(n)",
      "code_snippet": """class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        m = val if not self.min_stack else min(val, self.min_stack[-1])
        self.min_stack.append(m)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def getMin(self):
        return self.min_stack[-1]""",
      "language": "Python",
      "keywords": ["min stack", "O(1) minimum", "stack", "auxiliary stack"]}),

    ("Trie insertion and search run in O(m) where m is the key length, independent of the number of stored keys.",
     "data_structures", 0.95, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(m) insert and search",
      "space_complexity": "O(n * m) worst case",
      "code_snippet": """class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end""",
      "language": "Python",
      "keywords": ["trie", "prefix tree", "insert", "search", "autocomplete"]}),

    # ── Design pattern code snippets ───────────────────────────────────────────
    ("Observer pattern: subjects maintain a list of observers and notify them of state changes; decouples event producers from consumers.",
     "design_patterns", 0.95, KnowledgeType.DESIGN_PATTERN,
     {"code_snippet": """class Subject:
    def __init__(self):
        self._observers = []
        self._state = None

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self):
        for obs in self._observers:
            obs.update(self._state)

    def set_state(self, state):
        self._state = state
        self.notify()

class ConcreteObserver:
    def update(self, state):
        print(f'Observer received: {state}')""",
      "language": "Python",
      "keywords": ["observer", "subject", "notify", "event", "design pattern"]}),

    ("Strategy pattern encapsulates a family of algorithms and makes them interchangeable without changing the client.",
     "design_patterns", 0.94, KnowledgeType.DESIGN_PATTERN,
     {"code_snippet": """from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data): ...

class QuickSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # placeholder

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort(self, data):
        return self._strategy.sort(data)""",
      "language": "Python",
      "keywords": ["strategy pattern", "algorithm family", "interchangeable", "design pattern"]}),

    ("Decorator pattern wraps an object to add behaviour without modifying the wrapped object's class.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"code_snippet": """class TextComponent:
    def render(self): return 'Hello'

class BoldDecorator:
    def __init__(self, component):
        self._component = component
    def render(self):
        return f'<b>{self._component.render()}</b>'

class ItalicDecorator:
    def __init__(self, component):
        self._component = component
    def render(self):
        return f'<i>{self._component.render()}</i>'

text = ItalicDecorator(BoldDecorator(TextComponent()))
print(text.render())  # <i><b>Hello</b></i>""",
      "language": "Python",
      "keywords": ["decorator pattern", "wrapper", "composition", "design pattern"]}),

    ("Command pattern encapsulates a request as an object, supporting undo, queuing, and logging of operations.",
     "design_patterns", 0.92, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["command pattern", "undo", "queue", "macro", "design pattern"]}),

    ("Factory method pattern lets subclasses decide which class to instantiate; decouples client code from concrete product classes.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["factory method", "creation", "subclass", "product", "design pattern"]}),

    ("Abstract factory pattern provides an interface for creating families of related objects without specifying their concrete classes.",
     "design_patterns", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["abstract factory", "product family", "creation", "design pattern"]}),

    ("Builder pattern separates the construction of a complex object from its representation, enabling the same construction process to produce different representations.",
     "design_patterns", 0.92, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["builder pattern", "step-by-step", "fluent interface", "design pattern"]}),

    # ── Concurrency (more) ────────────────────────────────────────────────────
    ("Structured concurrency (Python TaskGroup, Java StructuredTaskScope) ensures child tasks are always awaited before the parent scope exits.",
     "concurrency", 0.89, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Python 3.11+ asyncio.TaskGroup; avoids fire-and-forget task leaks",
      "keywords": ["structured concurrency", "TaskGroup", "asyncio", "Python 3.11"]}),

    ("Reactive programming (RxJS, Project Reactor, RxJava) models asynchronous data streams with composable operators.",
     "concurrency", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["reactive programming", "RxJS", "Reactor", "Observable", "backpressure"]}),

    ("The producer-consumer pattern decouples data producers from consumers via a bounded buffer; use when production and consumption rates differ.",
     "concurrency", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"keywords": ["producer-consumer", "bounded buffer", "queue", "concurrency"]}),

    # ── Operating systems / systems programming ────────────────────────────────
    ("Copy-on-write (COW) defers copying until a write occurs; used in fork(), persistent data structures, and modern storage engines.",
     "systems", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["copy-on-write", "COW", "fork", "persistent data structure", "efficient"]}),

    ("mmap (memory-mapped files) allows file contents to be accessed as memory; the OS handles I/O lazily via page faults.",
     "systems", 0.89, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Large file random-access; inter-process shared memory; LMDB, SQLite WAL",
      "keywords": ["mmap", "memory-mapped file", "page fault", "IPC"]}),

    ("epoll / kqueue / io_uring allow a single thread to monitor thousands of file descriptors efficiently — the basis of high-performance async servers.",
     "systems", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["epoll", "kqueue", "io_uring", "async I/O", "event-driven", "Nginx"]}),

    ("Zero-copy I/O (sendfile, splice) transfers data between file descriptors in kernel space, bypassing user-space copying.",
     "systems", 0.89, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["zero-copy", "sendfile", "splice", "kernel", "performance"]}),

    # ── Cloud native (more) ───────────────────────────────────────────────────
    ("GitOps: use Git as the single source of truth for cluster state; reconciliation controllers (Argo CD, Flux) apply manifests automatically.",
     "cloud", 0.90, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["GitOps", "Argo CD", "Flux", "Kubernetes", "declarative"]}),

    ("Immutable infrastructure: never mutate running servers; replace them with newly provisioned ones on each deployment.",
     "cloud", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["immutable infrastructure", "golden image", "cattle not pets", "Phoenix server"]}),

    ("Object storage (S3, GCS) provides essentially unlimited, durable, and cheap storage for blobs; not suitable for random byte-range writes.",
     "cloud", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["object storage", "S3", "GCS", "blob", "durable"]}),

    ("Content-addressable storage identifies data by its cryptographic hash; identical content is stored only once (deduplication).",
     "cloud", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["content-addressable", "CAS", "hash", "deduplication", "Git objects"]}),

    # ── Math / complexity ─────────────────────────────────────────────────────
    ("Master theorem: T(n) = aT(n/b) + f(n); three cases determine whether f(n) dominates, the recursive calls dominate, or they are equal.",
     "algorithms", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["master theorem", "recurrence relation", "divide and conquer", "complexity"]}),

    ("NP-complete problems have no known polynomial algorithm; for practice, use heuristics, approximation algorithms, or FPT algorithms.",
     "algorithms", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["NP-complete", "NP-hard", "heuristic", "approximation", "complexity"]}),

    ("Amortized O(1) push in a dynamic array: double capacity on overflow; total work for n pushes is O(n) so amortized cost is O(1).",
     "data_structures", 0.95, KnowledgeType.ALGORITHM,
     {"keywords": ["dynamic array", "amortized", "doubling", "ArrayList", "vector"]}),

    # ── Numerical / scientific ────────────────────────────────────────────────
    ("Floating-point arithmetic is not associative; accumulate small numbers first and use compensated summation (Kahan) for accuracy.",
     "best_practices", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["floating point", "Kahan summation", "numerical stability", "IEEE 754"]}),

    ("IEEE 754 double precision has 15-17 significant decimal digits; use decimal.Decimal or integer arithmetic for exact currency computations.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["IEEE 754", "floating point", "currency", "Decimal", "precision"]}),

    # ── Miscellaneous best practices ──────────────────────────────────────────
    ("Document the 'why', not the 'what'; code explains what it does, comments explain why it does it that way.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["comments", "documentation", "why not what", "readability"]}),

    ("Incremental commits with meaningful messages make debugging with git bisect and git log faster and more reliable.",
     "best_practices", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["git commit", "commit message", "git bisect", "history"]}),

    ("Feature branches and pull requests enable code review and CI checks before merging, preventing broken code on main.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["feature branch", "pull request", "code review", "CI", "GitHub flow"]}),

    ("The Boy Scout Rule: leave the codebase cleaner than you found it; fix small issues during routine work.",
     "best_practices", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["Boy Scout Rule", "refactoring", "technical debt", "clean code"]}),

    ("Rubber duck debugging: explaining code step-by-step to an imaginary listener often reveals the bug without external help.",
     "best_practices", 0.88, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["rubber duck debugging", "self-explanation", "debugging"]}),

    ("The principle of least astonishment: a component should behave in a way consistent with user/developer expectations.",
     "best_practices", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["principle of least astonishment", "API design", "UX", "consistency"]}),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def seed_extra_axioms(kg: KnowledgeGraph) -> int:
    """Seed EXTRA_AXIOMS directly using the same pattern as CodeSeeder.seed_code_axioms."""
    from tqdm import tqdm
    created = 0
    for prop, domain, conf, ktype, extras in tqdm(EXTRA_AXIOMS, desc="Extra axioms"):
        eu = EpistemicUnit(
            proposition=prop,
            knowledge_type=ktype,
            confidence=conf,
            domain=domain,
            sources=[Source(name="SSM Expanded Knowledge Base", reliability_score=0.95)],
            keywords=extras.get("keywords", []),
            epistemic_gaps=[],
            code_snippet=extras.get("code_snippet"),
            language=extras.get("language"),
            time_complexity=extras.get("time_complexity"),
            space_complexity=extras.get("space_complexity"),
            when_to_use=extras.get("when_to_use"),
            tradeoffs=extras.get("tradeoffs"),
        )
        if kg.add(eu):
            created += 1
    print(f"[Extra] Seeded {created} extra axioms / best-practices / patterns")
    return created


def rebuild_faiss(kg: KnowledgeGraph, data_path: str) -> None:
    """
    Re-embed all EU propositions and build a fresh FAISS IndexFlatL2.
    Saves faiss.index and faiss_map.json to data_path.
    Skips EUs that already have embeddings (uses existing vector).
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\n[FAISS] Rebuilding index for {len(kg.units)} units...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = 384

    all_ids: list[str] = []
    all_vecs: list[list[float]] = []

    units_list = list(kg.units.values())
    batch_size = 256

    for i in range(0, len(units_list), batch_size):
        batch = units_list[i : i + batch_size]
        propositions = [eu.proposition for eu in batch]
        embeddings = model.encode(propositions, show_progress_bar=False)
        for eu, emb in zip(batch, embeddings):
            eu.embedding = emb.tolist()
            all_ids.append(eu.id)
            all_vecs.append(emb.tolist())
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {min(i + batch_size, len(units_list))}/{len(units_list)}")

    vecs_np = np.array(all_vecs, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(vecs_np)

    faiss.write_index(index, f"{data_path}/faiss.index")
    with open(f"{data_path}/faiss_map.json", "w") as f:
        json.dump(all_ids, f)

    # Sync kg internals
    kg._faiss_index = index
    kg.faiss_id_map = all_ids

    print(f"[FAISS] Saved {index.ntotal} vectors → {data_path}/faiss.index")


def upload_to_hf(data_path: str, repo_id: str) -> None:
    """Upload units.json, faiss.index, faiss_map.json to HF dataset under epichat/."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = ["units.json", "faiss.index", "faiss_map.json"]

    print(f"\n[HF] Uploading to {repo_id} / epichat/ ...")
    for fname in files:
        local = os.path.join(data_path, fname)
        if not os.path.exists(local):
            print(f"  [HF] Skipping {fname} (not found)")
            continue
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=f"epichat/{fname}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  [HF] Uploaded epichat/{fname}")

    print("[HF] Upload complete.")


def main() -> None:
    t0 = time.time()

    # ── Step 1: Load existing KG ──────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading existing knowledge graph")
    print("=" * 60)
    kg = KnowledgeGraph()
    loaded = kg.load(EPISTEME_DATA)
    initial_count = len(kg.units)
    print(f"Loaded {initial_count} existing units.\n")

    seeder = CodeSeeder(kg)

    # ── Step 2: Wikipedia seeding ─────────────────────────────────────────────
    print("=" * 60)
    print(f"Step 2: Wikipedia seeding ({len(WIKI_TOPICS)} articles, max_sentences=50)")
    print("=" * 60)
    wiki_added = 0
    try:
        wiki_added = seeder.seed_wikipedia(topics=WIKI_TOPICS, max_sentences=50)
        print(f"[Wiki] Added {wiki_added} units.  Total now: {len(kg.units)}\n")
    except Exception as exc:
        print(f"[Wiki] ERROR during Wikipedia seeding: {exc}")
        import traceback; traceback.print_exc()

    # ── Step 3: Extra hand-crafted axioms ─────────────────────────────────────
    print("=" * 60)
    print(f"Step 3: Extra axioms ({len(EXTRA_AXIOMS)} entries)")
    print("=" * 60)
    extra_added = 0
    try:
        extra_added = seed_extra_axioms(kg)
        print(f"[Extra] Added {extra_added} units.  Total now: {len(kg.units)}\n")
    except Exception as exc:
        print(f"[Extra] ERROR during extra axiom seeding: {exc}")
        import traceback; traceback.print_exc()

    # ── Step 4: Save KG ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 4: Saving knowledge graph")
    print("=" * 60)
    try:
        kg.save(EPISTEME_DATA)
    except Exception as exc:
        print(f"[Save] ERROR: {exc}")
        import traceback; traceback.print_exc()

    # ── Step 5: Rebuild FAISS index ───────────────────────────────────────────
    print("=" * 60)
    print("Step 5: Rebuilding FAISS index")
    print("=" * 60)
    try:
        rebuild_faiss(kg, EPISTEME_DATA)
        # Re-save after embeddings updated
        kg.save(EPISTEME_DATA)
    except Exception as exc:
        print(f"[FAISS] ERROR: {exc}")
        import traceback; traceback.print_exc()

    # ── Step 6: Upload to Hugging Face ────────────────────────────────────────
    print("=" * 60)
    print(f"Step 6: Uploading to HF dataset {HF_REPO}")
    print("=" * 60)
    try:
        upload_to_hf(EPISTEME_DATA, HF_REPO)
    except Exception as exc:
        print(f"[HF] ERROR: {exc}")
        import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    final_count = len(kg.units)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Initial units:       {initial_count:>8,}")
    print(f"  Wikipedia added:     {wiki_added:>8,}")
    print(f"  Extra axioms added:  {extra_added:>8,}")
    print(f"  Final unit count:    {final_count:>8,}")
    print(f"  Net new units:       {final_count - initial_count:>8,}")
    print(f"  Elapsed:             {elapsed/60:.1f} min")
    print("=" * 60)
    stats = kg.stats()
    print(f"  Avg confidence:      {stats['avg_confidence']:.3f}")
    print(f"  Domains:             {len(stats['domains'])}")
    for domain, count in sorted(stats["domains"].items(), key=lambda x: -x[1])[:15]:
        print(f"    {domain:<30} {count:>6,}")
    if len(stats["domains"]) > 15:
        print(f"    ... and {len(stats['domains'])-15} more domains")
    print("=" * 60)


if __name__ == "__main__":
    main()
