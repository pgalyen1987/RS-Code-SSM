"""
Mass EU generation for EpiChat — targets 15,000+ EpistemicUnits.

Coverage:
  Python, TypeScript, Kotlin, Solidity, Java, Rust, Go, C++
  Databases (PostgreSQL, MySQL, MongoDB, Redis, Cassandra, ClickHouse, Neo4j, TimescaleDB)
  Networking (TCP/IP, DNS, HTTP, TLS, BGP, routing, firewalls, WebRTC, QUIC)
  Algorithms, Data Structures, System Design, Security, ML, Cloud

Three-pronged approach:
  1. Wikipedia scraping: ~500 topics × 25 sentences ≈ 4000+ EUs
  2. LLM synthesis:      Ollama generates structured EUs with code snippets per language
  3. Dedup & save every 50 new EUs

Run overnight:
    source scripts/env.sh  # or set EPICHAT_DIR, REPO_ROOT
    python scripts/generate_eus.py 2>&1 | tee logs/eu_expansion.log
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# epichat is in-repo at REPO_ROOT/epichat — add REPO_ROOT so "epichat" is importable
_repo = Path(__file__).resolve().parent.parent
REPO_ROOT = os.environ.get("REPO_ROOT") or str(_repo)
EPICHAT_DIR = os.environ.get("EPICHAT_DIR") or str(Path(REPO_ROOT) / "epichat")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from epichat.core.knowledge_graph import KnowledgeGraph
from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from epichat.seeding.code_seeder import CodeSeeder


# ─── 1. Wikipedia topics (~500 topics) ───────────────────────────────────────

WIKI_TOPICS = [
    # ── Core CS ──────────────────────────────────────────────────────────────
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

    # ── OOP & Patterns ────────────────────────────────────────────────────────
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

    # ── ML & AI ───────────────────────────────────────────────────────────────
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

    # ── Databases — broad ─────────────────────────────────────────────────────
    "Database", "Relational database", "SQL", "NoSQL",
    "Database index", "Database normalization", "ACID",
    "Database transaction", "Query optimization", "PostgreSQL",
    "MySQL", "SQLite", "MariaDB", "Oracle Database",
    "MongoDB", "Redis", "Elasticsearch", "Apache Solr",
    "Apache Cassandra", "Apache HBase", "Apache Druid",
    "ClickHouse", "InfluxDB", "TimescaleDB", "QuestDB",
    "Graph database", "Neo4j", "Amazon DynamoDB",
    "CAP theorem", "PACELC theorem",
    "Eventual consistency", "Database sharding", "Replication (computing)",
    "Two-phase commit protocol", "Write-ahead logging",
    "Multiversion concurrency control", "Object–relational mapping",
    "Stored procedure", "Trigger (databases)", "Database view",
    "Materialized view", "Database cursor", "Database partitioning",
    "Database connection pool", "Database lock",
    "Isolation (database systems)", "Database deadlock",
    "Column-oriented DBMS", "Row-oriented DBMS",
    "In-memory database", "Embedded database",
    "Full-text search", "Inverted index",
    "Data warehouse", "OLAP", "OLTP",
    "Extract, transform, load", "Data lake",
    "Apache Spark", "Apache Flink", "Apache Beam",
    "Snowflake (data platform)", "Google BigQuery",

    # ── Networking — broad ────────────────────────────────────────────────────
    "Computer network", "OSI model", "Internet protocol suite",
    "Transmission Control Protocol", "User Datagram Protocol",
    "Internet Protocol", "IPv4", "IPv6",
    "Domain Name System", "Dynamic Host Configuration Protocol",
    "Network address translation", "Firewall (computing)",
    "Virtual private network", "Proxy server", "Reverse proxy",
    "Load balancing (computing)", "Content delivery network",
    "Border Gateway Protocol", "Open Shortest Path First",
    "Routing Information Protocol", "Routing table",
    "Ethernet", "Wi-Fi", "Network switch", "Network router",
    "Hypertext Transfer Protocol", "Hypertext Transfer Protocol Secure",
    "HTTP/2", "HTTP/3", "QUIC", "WebSocket",
    "Transport Layer Security", "Datagram Transport Layer Security",
    "Network packet", "IP address", "Classless Inter-Domain Routing",
    "Subnet mask", "Default gateway", "MAC address",
    "Address Resolution Protocol", "Network Time Protocol",
    "Simple Mail Transfer Protocol", "File Transfer Protocol",
    "Secure Shell", "Remote Desktop Protocol",
    "Network topology", "Peer-to-peer", "Client–server model",
    "Software-defined networking", "Network functions virtualization",
    "Anycast", "Multicast", "Broadcast (networking)",
    "Bandwidth (computing)", "Latency (engineering)",
    "Network congestion", "Quality of service",
    "Deep packet inspection", "Intrusion detection system",
    "Denial-of-service attack", "Distributed denial-of-service attack",
    "Man-in-the-middle attack", "Packet sniffing",
    "WebRTC", "STUN", "TURN", "ICE (networking)",
    "MQTT", "AMQP", "gRPC",

    # ── C ─────────────────────────────────────────────────────────────────────
    "C (programming language)", "C standard library", "Pointer (computer programming)",
    "Memory management", "Dynamic memory allocation", "Buffer overflow",
    "Undefined behavior", "C preprocessor", "Header file",
    "Struct (C programming language)", "Union type", "Bit field",
    "File descriptor", "POSIX", "C POSIX library",
    "setjmp.h", "Signal (IPC)", "Inter-process communication",

    # ── C++ ───────────────────────────────────────────────────────────────────
    "C++", "C++ Standard Library", "Standard Template Library",
    "Smart pointer", "RAII", "Move semantics",
    "C++ classes", "Virtual function", "Multiple inheritance",
    "Template (C++)", "C++ concepts", "Lambda expression (C++)",
    "C++11", "C++17", "C++20",
    "Boost (C++ libraries)", "CMake",

    # ── Java ──────────────────────────────────────────────────────────────────
    "Java (programming language)", "Java virtual machine", "Java Class Library",
    "Java collections framework", "Java generics", "Java reflection",
    "Java annotation", "Java concurrency", "Java memory model",
    "Java garbage collection", "Java stream", "Optional (Java)",
    "Spring Framework", "Hibernate (framework)", "Maven",
    "Gradle", "Java module system",

    # ── C# ────────────────────────────────────────────────────────────────────
    "C Sharp (programming language)", ".NET", "Common Language Runtime",
    "Language Integrated Query", "Async/await", "Task (computing)",
    "C# generics", "Delegate (CLI)", "C# extension method",
    "Entity Framework", "ASP.NET Core", "NuGet",
    "C# record type", "C# pattern matching", "Nullable reference type",

    # ── TypeScript ────────────────────────────────────────────────────────────
    "TypeScript", "Type system", "Type inference",
    "Generic programming", "Structural type system",
    "Nominal type system", "Covariance and contravariance (computer science)",
    "Decorator pattern", "ECMAScript", "JavaScript",
    "Node.js", "Deno (software)", "Bun (software)",

    # ── Kotlin ────────────────────────────────────────────────────────────────
    "Kotlin (programming language)", "Coroutine", "Extension method",
    "Null safety", "Sealed class", "Data class",
    "Kotlin (programming language)", "Android (operating system)",
    "JetBrains", "JVM", "Kotlin Multiplatform",
    "Ktor", "Spring Framework",

    # ── Solidity & Blockchain ─────────────────────────────────────────────────
    "Solidity", "Ethereum", "Smart contract", "Blockchain",
    "Ethereum Virtual Machine", "Gas (Ethereum)",
    "Cryptocurrency", "Decentralized finance",
    "Non-fungible token", "Decentralized autonomous organization",
    "Proof of work", "Proof of stake", "Merkle tree",
    "Cryptographic hash function", "Digital signature",
    "Elliptic-curve cryptography", "Zero-knowledge proof",
    "Layer 2 blockchain solutions", "Rollup (blockchain)",
    "Oracle (blockchain)", "Chainlink",
    "OpenZeppelin", "ERC-20", "ERC-721",
    "Hardhat (software)", "Foundry (Ethereum)",
    "Uniswap", "Aave", "Compound Finance",

    # ── Distributed Systems ───────────────────────────────────────────────────
    "Distributed computing", "Consensus (computer science)", "Raft (algorithm)",
    "Paxos (computer science)", "MapReduce", "Apache Kafka",
    "Message queue", "Remote procedure call",
    "Distributed hash table", "Consistent hashing",
    "Byzantine fault tolerance", "Leader election",
    "Service discovery", "Apache ZooKeeper",
    "Service mesh", "Istio", "Envoy (software)",

    # ── Systems & OS ──────────────────────────────────────────────────────────
    "Operating system", "Process (computing)", "Thread (computing)",
    "Concurrency (computer science)", "Deadlock", "Semaphore (programming)",
    "Mutex", "Memory management", "Virtual memory", "Cache (computing)",
    "CPU cache", "Garbage collection (computer science)", "Compiler",
    "Just-in-time compilation", "Interpreter (computing)",
    "Abstract syntax tree", "Parsing", "Lexical analysis",
    "Context switch", "Scheduling (computing)", "Memory paging",
    "Copy-on-write", "Memory-mapped file",
    "NUMA", "False sharing",

    # ── Languages & Paradigms ─────────────────────────────────────────────────
    "Python (programming language)", "Java (programming language)",
    "Rust (programming language)", "Go (programming language)",
    "C++ (programming language)", "Functional programming",
    "Metaprogramming", "Reflection (computer programming)",
    "Continuation-passing style", "Monads (functional programming)",
    "Lazy evaluation", "Currying", "Higher-order function",
    "Closure (computer programming)", "First-class function",
    "Async/await", "Event loop",

    # ── Web & Cloud ───────────────────────────────────────────────────────────
    "REST", "GraphQL", "OAuth", "JSON Web Token",
    "Microservices", "API gateway",
    "Docker (software)", "Kubernetes", "Continuous integration",
    "Continuous deployment", "Infrastructure as code",
    "Serverless computing", "Cloud computing",
    "Circuit breaker (computing)", "Rate limiting",
    "React (software)", "Vue.js", "Angular (web framework)",
    "Next.js", "NestJS",

    # ── Security ──────────────────────────────────────────────────────────────
    "Computer security", "OWASP", "SQL injection",
    "Cross-site scripting", "Cross-site request forgery",
    "Public-key cryptography", "Symmetric-key algorithm",
    "Transport Layer Security", "Authentication", "Authorization",
    "Penetration testing", "Firewall (computing)",
    "Buffer overflow", "Race condition", "Zero-day vulnerability",
    "Intrusion detection system",
    "Hash function", "SHA-2", "Bcrypt",

    # ── Architecture ──────────────────────────────────────────────────────────
    "Software architecture", "Model–view–controller", "CQRS",
    "Event-driven architecture", "Hexagonal architecture",
    "Domain-driven design", "Event sourcing",

    # ── Testing ───────────────────────────────────────────────────────────────
    "Test-driven development", "Unit testing", "Integration testing",
    "Behavior-driven development", "Property-based testing",
    "Fuzzing", "Mutation testing", "Code coverage",

    # ── Tools & Practices ─────────────────────────────────────────────────────
    "Version control", "Git", "Refactoring", "Technical debt",
    "Code review", "Clean code", "Cyclomatic complexity",
]


# ─── 2. LLM-synthesized EU topics ─────────────────────────────────────────────

LLM_CONCEPTS = [
    # ── Python ────────────────────────────────────────────────────────────────
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
    ("Python abstract base classes and the abc module", "python"),
    ("Python Protocol class for structural subtyping", "python"),
    ("Python __dunder__ methods and operator overloading", "python"),
    ("Python multiprocessing Pool for CPU-bound parallelism", "python"),
    ("Python subprocess module for running shell commands", "python"),
    ("Python __enter__ and __exit__ for custom context managers", "python"),
    ("Python enum.Enum for typed constants", "python"),
    ("Python importlib for dynamic module loading", "python"),
    ("Python weakref for cache-friendly object references", "python"),
    ("Python struct module for binary data packing", "python"),
    ("Python ctypes for calling C libraries from Python", "python"),
    ("Python Pydantic for data validation and settings management", "python"),
    ("Python FastAPI async request handling and dependency injection", "python"),
    ("Python SQLAlchemy ORM: session, query, relationship", "python"),
    ("Python argparse for CLI argument parsing", "python"),
    ("Python packaging: pyproject.toml, setup.cfg, wheels", "python"),
    ("Python __all__ for controlling module exports", "python"),
    ("Python atexit and signal handlers for graceful shutdown", "python"),

    # ── C ─────────────────────────────────────────────────────────────────────
    ("C pointers: pointer arithmetic, pointer to pointer, void*", "c"),
    ("C memory management: malloc, calloc, realloc, free", "c"),
    ("C stack vs heap allocation and when to use each", "c"),
    ("C undefined behavior: common causes and consequences", "c"),
    ("C string handling: strcpy, strncpy, snprintf safety", "c"),
    ("C struct padding and memory alignment", "c"),
    ("C function pointers for callbacks and dispatch tables", "c"),
    ("C preprocessor macros: #define, #ifdef, #include guards", "c"),
    ("C variadic functions with stdarg.h (va_list, va_arg)", "c"),
    ("C file I/O: fopen, fread, fwrite, fseek, fclose", "c"),
    ("C POSIX threads: pthread_create, mutex, condition variables", "c"),
    ("C signal handling: SIGINT, SIGTERM, signal(), sigaction()", "c"),
    ("C bit manipulation: masks, shifts, bitfields", "c"),
    ("C unions for type-punning and memory reuse", "c"),
    ("C linked list implementation with struct and pointers", "c"),
    ("C buffer overflow: stack smashing, canaries, ASLR protection", "c"),
    ("C setjmp/longjmp for non-local error handling", "c"),
    ("C inline assembly for performance-critical sections", "c"),
    ("C restrict keyword for pointer aliasing optimization", "c"),
    ("C11 atomics: _Atomic, atomic_fetch_add, memory ordering", "c"),
    ("C interop with Python via ctypes and CFFI", "c"),
    ("C Makefile: targets, variables, pattern rules, phony targets", "c"),
    ("C valgrind for memory leak and error detection", "c"),
    ("C address sanitizer (ASan) for runtime memory error detection", "c"),
    ("C static analysis with clang-tidy and cppcheck", "c"),

    # ── C++ ───────────────────────────────────────────────────────────────────
    ("C++ RAII: resource acquisition is initialization pattern", "cpp"),
    ("C++ smart pointers: unique_ptr, shared_ptr, weak_ptr", "cpp"),
    ("C++ move semantics: rvalue references, std::move, move constructor", "cpp"),
    ("C++ perfect forwarding with std::forward and universal references", "cpp"),
    ("C++ template metaprogramming and SFINAE", "cpp"),
    ("C++ concepts (C++20) for constraining template parameters", "cpp"),
    ("C++ variadic templates and parameter packs", "cpp"),
    ("C++ lambda expressions: capture by value vs reference", "cpp"),
    ("C++ STL containers: vector, deque, list, map, unordered_map", "cpp"),
    ("C++ STL algorithms: sort, find_if, transform, accumulate", "cpp"),
    ("C++ iterators: input, output, forward, bidirectional, random access", "cpp"),
    ("C++ virtual functions, vtable, and dynamic dispatch", "cpp"),
    ("C++ multiple inheritance and diamond problem with virtual base", "cpp"),
    ("C++ rule of zero, three, five for resource management", "cpp"),
    ("C++ const correctness: const methods, const references", "cpp"),
    ("C++ operator overloading: arithmetic, comparison, stream operators", "cpp"),
    ("C++ std::optional for nullable values without pointers", "cpp"),
    ("C++ std::variant for type-safe unions (C++17)", "cpp"),
    ("C++ std::string_view for zero-copy string references", "cpp"),
    ("C++ structured bindings for tuple and struct unpacking (C++17)", "cpp"),
    ("C++ ranges library for composable lazy sequences (C++20)", "cpp"),
    ("C++ coroutines: co_await, co_yield, co_return (C++20)", "cpp"),
    ("C++ memory model: acquire-release, sequential consistency", "cpp"),
    ("C++ std::thread, std::mutex, std::condition_variable", "cpp"),
    ("C++ std::atomic for lock-free concurrent programming", "cpp"),
    ("C++ exception handling: try, catch, throw, noexcept", "cpp"),
    ("C++ CRTP (Curiously Recurring Template Pattern) for static polymorphism", "cpp"),
    ("C++ pimpl idiom for ABI stability and compilation firewalls", "cpp"),
    ("C++ copy elision and return value optimization (RVO/NRVO)", "cpp"),
    ("C++ constexpr and compile-time computation", "cpp"),

    # ── Java ──────────────────────────────────────────────────────────────────
    ("Java generics: type erasure, bounded wildcards, PECS", "java"),
    ("Java collections: ArrayList vs LinkedList vs ArrayDeque", "java"),
    ("Java HashMap internals: hashing, load factor, rehashing", "java"),
    ("Java streams: filter, map, flatMap, collect, reduce", "java"),
    ("Java Optional to avoid NullPointerException", "java"),
    ("Java functional interfaces: Predicate, Function, Consumer, Supplier", "java"),
    ("Java CompletableFuture for async composition", "java"),
    ("Java virtual threads (Project Loom, Java 21)", "java"),
    ("Java memory model: happens-before, volatile, synchronized", "java"),
    ("Java garbage collection: G1GC, ZGC, Shenandoah", "java"),
    ("Java reflection: Class, Method, Field, Proxy", "java"),
    ("Java annotations: retention, target, custom annotation processors", "java"),
    ("Java equals and hashCode contract and common mistakes", "java"),
    ("Java Comparable vs Comparator for sorting", "java"),
    ("Java record classes for immutable data (Java 16+)", "java"),
    ("Java sealed classes for exhaustive pattern matching (Java 17+)", "java"),
    ("Java pattern matching with instanceof and switch (Java 21)", "java"),
    ("Java module system (JPMS): module-info.java, requires, exports", "java"),
    ("Java serialization pitfalls and alternatives (JSON, Protobuf)", "java"),
    ("Java checked vs unchecked exceptions: when to use each", "java"),
    ("Java try-with-resources for AutoCloseable", "java"),
    ("Java concurrency: ExecutorService, thread pools, ForkJoinPool", "java"),
    ("Java BlockingQueue for producer-consumer patterns", "java"),
    ("Java ConcurrentHashMap vs Collections.synchronizedMap", "java"),
    ("Java reactive programming with Project Reactor (Flux, Mono)", "java"),
    ("Spring Boot autoconfiguration and starter dependencies", "java"),
    ("Spring dependency injection: @Component, @Bean, @Autowired", "java"),
    ("Spring Data JPA: repository pattern, JPQL, named queries", "java"),
    ("JVM JIT compilation: tiered compilation, C1 vs C2 compilers", "java"),
    ("Java profiling with JFR (Java Flight Recorder) and async-profiler", "java"),

    # ── C# ────────────────────────────────────────────────────────────────────
    ("C# async/await: Task, ValueTask, ConfigureAwait(false)", "csharp"),
    ("C# LINQ: deferred execution, IEnumerable vs IQueryable", "csharp"),
    ("C# generics: constraints, covariance, contravariance (in/out)", "csharp"),
    ("C# delegates, Func, Action, Predicate, and events", "csharp"),
    ("C# extension methods for adding behavior without inheritance", "csharp"),
    ("C# record types for immutable data with value equality", "csharp"),
    ("C# pattern matching: switch expressions, property patterns, list patterns", "csharp"),
    ("C# nullable reference types and null-state analysis", "csharp"),
    ("C# span<T> and memory<T> for zero-allocation slicing", "csharp"),
    ("C# ref structs and stack-only types for performance", "csharp"),
    ("C# IDisposable and the dispose pattern for resource cleanup", "csharp"),
    ("C# dependency injection with Microsoft.Extensions.DependencyInjection", "csharp"),
    ("C# Entity Framework Core: migrations, DbContext, lazy loading", "csharp"),
    ("C# ASP.NET Core middleware pipeline and request handling", "csharp"),
    ("C# minimal API vs controller-based API in ASP.NET Core", "csharp"),
    ("C# channels for producer-consumer with System.Threading.Channels", "csharp"),
    ("C# CancellationToken for cooperative task cancellation", "csharp"),
    ("C# SemaphoreSlim and AsyncLock for async concurrency control", "csharp"),
    ("C# source generators for compile-time code generation", "csharp"),
    ("C# reflection and dynamic code with Expression trees", "csharp"),
    ("C# unsafe code and fixed statement for pointer operations", "csharp"),
    ("C# struct vs class: value vs reference semantics", "csharp"),
    ("C# interface default implementations (C# 8+)", "csharp"),
    ("C# global using and implicit usings (C# 10+)", "csharp"),
    ("C# required members and primary constructors (C# 12)", "csharp"),
    ("C# xUnit and NUnit testing patterns with mock frameworks", "csharp"),
    ("C# BenchmarkDotNet for micro-benchmarking", "csharp"),
    ("C# SignalR for real-time WebSocket communication", "csharp"),
    ("C# Polly for resilience: retry, circuit breaker, fallback", "csharp"),
    ("C# MediatR for CQRS and mediator pattern", "csharp"),

    # ── TypeScript ────────────────────────────────────────────────────────────
    ("TypeScript strict mode and why it matters", "typescript"),
    ("TypeScript union types and type narrowing with guards", "typescript"),
    ("TypeScript intersection types for object composition", "typescript"),
    ("TypeScript generic constraints and conditional types", "typescript"),
    ("TypeScript mapped types: Partial, Required, Readonly, Record", "typescript"),
    ("TypeScript template literal types for string manipulation", "typescript"),
    ("TypeScript discriminated unions for exhaustive pattern matching", "typescript"),
    ("TypeScript infer keyword in conditional types", "typescript"),
    ("TypeScript decorators: class, method, property, parameter", "typescript"),
    ("TypeScript utility types: Pick, Omit, Exclude, Extract", "typescript"),
    ("TypeScript keyof and typeof operators", "typescript"),
    ("TypeScript declaration merging and module augmentation", "typescript"),
    ("TypeScript satisfies operator vs type assertion vs annotation", "typescript"),
    ("TypeScript enums vs const enums vs union types", "typescript"),
    ("TypeScript function overloads for polymorphic APIs", "typescript"),
    ("TypeScript namespace vs ES modules", "typescript"),
    ("TypeScript strict null checks and non-null assertion", "typescript"),
    ("TypeScript index signatures and mapped object types", "typescript"),
    ("TypeScript readonly arrays and immutable data patterns", "typescript"),
    ("TypeScript branded types for nominal type safety", "typescript"),
    ("TypeScript ReturnType, Parameters, ConstructorParameters utilities", "typescript"),
    ("TypeScript async/await with proper error handling patterns", "typescript"),
    ("TypeScript interface vs type alias: when to use each", "typescript"),
    ("TypeScript module resolution: node vs bundler vs classic", "typescript"),
    ("TypeScript tsconfig.json key settings explained", "typescript"),
    ("TypeScript Zod for runtime type validation", "typescript"),
    ("TypeScript type guards with instanceof and custom predicates", "typescript"),
    ("TypeScript covariance and contravariance in function types", "typescript"),
    ("TypeScript declaration files (.d.ts) and DefinitelyTyped", "typescript"),
    ("TypeScript path aliases and barrel files for project structure", "typescript"),

    # ── Kotlin ────────────────────────────────────────────────────────────────
    ("Kotlin data classes: equals, hashCode, copy, destructuring", "kotlin"),
    ("Kotlin sealed classes for exhaustive when expressions", "kotlin"),
    ("Kotlin coroutines: suspend functions and coroutine builders", "kotlin"),
    ("Kotlin Flow for reactive streams", "kotlin"),
    ("Kotlin extension functions and extension properties", "kotlin"),
    ("Kotlin null safety: ?., ?:, !!, let, also, run, apply", "kotlin"),
    ("Kotlin companion objects vs object declarations", "kotlin"),
    ("Kotlin inline functions and reified type parameters", "kotlin"),
    ("Kotlin delegation: by lazy, by observable, interface delegation", "kotlin"),
    ("Kotlin higher-order functions and lambda syntax", "kotlin"),
    ("Kotlin data classes with copy for immutable updates", "kotlin"),
    ("Kotlin coroutine scopes: GlobalScope vs viewModelScope vs lifecycleScope", "kotlin"),
    ("Kotlin channels vs flows for inter-coroutine communication", "kotlin"),
    ("Kotlin structured concurrency with coroutineScope and supervisorScope", "kotlin"),
    ("Kotlin value classes (inline classes) for zero-overhead wrappers", "kotlin"),
    ("Kotlin type aliases for complex type signatures", "kotlin"),
    ("Kotlin destructuring declarations for pairs and data classes", "kotlin"),
    ("Kotlin operator overloading: plus, minus, invoke, get, set", "kotlin"),
    ("Kotlin scope functions: let, run, with, apply, also", "kotlin"),
    ("Kotlin collections: filter, map, flatMap, fold, groupBy", "kotlin"),
    ("Kotlin sequence for lazy collection processing", "kotlin"),
    ("Kotlin object expressions vs object declarations", "kotlin"),
    ("Kotlin Ktor for building HTTP servers", "kotlin"),
    ("Kotlin Multiplatform: sharing code between JVM, iOS, JS", "kotlin"),
    ("Kotlin Arrow for functional programming: Option, Either, Validated", "kotlin"),
    ("Kotlin coroutine exception handling with CoroutineExceptionHandler", "kotlin"),
    ("Kotlin StateFlow and SharedFlow for state management", "kotlin"),
    ("Kotlin contracts for smart casts with require and check", "kotlin"),

    # ── Solidity ──────────────────────────────────────────────────────────────
    ("Solidity storage vs memory vs calldata: gas costs and usage", "solidity"),
    ("Solidity reentrancy attack and checks-effects-interactions pattern", "solidity"),
    ("Solidity ERC-20 token standard: transfer, approve, allowance", "solidity"),
    ("Solidity ERC-721 NFT standard: mint, transfer, tokenURI", "solidity"),
    ("Solidity ERC-1155 multi-token standard for gas efficiency", "solidity"),
    ("Solidity access control: Ownable, AccessControl, role-based", "solidity"),
    ("Solidity gas optimization: packing structs, uint8 vs uint256", "solidity"),
    ("Solidity events and indexed parameters for off-chain indexing", "solidity"),
    ("Solidity custom errors vs require strings for gas savings", "solidity"),
    ("Solidity proxy patterns: transparent, UUPS, beacon proxy", "solidity"),
    ("Solidity delegatecall and the proxy storage layout problem", "solidity"),
    ("Solidity modifier for access control and preconditions", "solidity"),
    ("Solidity mapping vs array: when to use each", "solidity"),
    ("Solidity interface vs abstract contract vs library", "solidity"),
    ("Solidity SafeMath and overflow protection in Solidity 0.8+", "solidity"),
    ("Solidity msg.sender vs tx.origin: security implications", "solidity"),
    ("Solidity fallback and receive functions", "solidity"),
    ("Solidity CREATE2 for deterministic contract deployment", "solidity"),
    ("Solidity flash loan attacks and protection", "solidity"),
    ("Solidity oracle manipulation and price feed security", "solidity"),
    ("Solidity front-running: MEV and commit-reveal schemes", "solidity"),
    ("Solidity multi-sig wallet pattern", "solidity"),
    ("Solidity timelock for governance actions", "solidity"),
    ("Solidity assembly (Yul) for gas optimization", "solidity"),
    ("Solidity immutable vs constant variables", "solidity"),
    ("Solidity constructor vs initializer for upgradeable contracts", "solidity"),
    ("Solidity testing with Foundry: forge test, fuzzing, invariants", "solidity"),
    ("Solidity Hardhat: deployment scripts, tasks, plugins", "solidity"),
    ("Solidity ABI encoding and decoding with abi.encode", "solidity"),
    ("Solidity DeFi: AMM design and the constant product formula", "solidity"),

    # ── Databases — PostgreSQL deep ───────────────────────────────────────────
    ("PostgreSQL VACUUM and autovacuum for dead tuple cleanup", "databases"),
    ("PostgreSQL EXPLAIN ANALYZE for query plan interpretation", "databases"),
    ("PostgreSQL partial indexes for filtered queries", "databases"),
    ("PostgreSQL GIN index for JSONB, arrays, and full-text search", "databases"),
    ("PostgreSQL BRIN index for sequential data (timestamps, IDs)", "databases"),
    ("PostgreSQL table partitioning: range, list, hash", "databases"),
    ("PostgreSQL MVCC: how concurrent transactions see data", "databases"),
    ("PostgreSQL WAL (Write-Ahead Log) for durability and replication", "databases"),
    ("PostgreSQL logical replication and publication/subscription", "databases"),
    ("PostgreSQL row-level security for multi-tenant applications", "databases"),
    ("PostgreSQL JSONB operators and indexing strategies", "databases"),
    ("PostgreSQL full-text search with tsvector and tsquery", "databases"),
    ("PostgreSQL window functions: ROW_NUMBER, RANK, LEAD, LAG", "databases"),
    ("PostgreSQL CTEs and recursive queries", "databases"),
    ("PostgreSQL lateral joins for correlated subqueries", "databases"),
    ("PostgreSQL advisory locks for application-level locking", "databases"),
    ("PostgreSQL pg_stat_statements for query performance monitoring", "databases"),
    ("PostgreSQL connection pooling with PgBouncer", "databases"),
    ("PostgreSQL tablespaces and storage management", "databases"),
    ("PostgreSQL extensions: pg_trgm, uuid-ossp, PostGIS", "databases"),

    # ── Databases — MySQL ─────────────────────────────────────────────────────
    ("MySQL InnoDB vs MyISAM: ACID compliance and locking", "databases"),
    ("MySQL binary log for replication and point-in-time recovery", "databases"),
    ("MySQL query cache vs buffer pool tuning", "databases"),
    ("MySQL composite indexes and index prefix length", "databases"),
    ("MySQL EXPLAIN output: type, key, rows, Extra fields", "databases"),
    ("MySQL transaction isolation levels and phantom reads", "databases"),
    ("MySQL GROUP_CONCAT and JSON aggregation functions", "databases"),

    # ── Databases — Redis ─────────────────────────────────────────────────────
    ("Redis data types: string, list, set, sorted set, hash, stream", "databases"),
    ("Redis persistence: RDB snapshots vs AOF append-only file", "databases"),
    ("Redis pub/sub for real-time messaging", "databases"),
    ("Redis streams for event sourcing and consumer groups", "databases"),
    ("Redis Lua scripting for atomic multi-key operations", "databases"),
    ("Redis cluster: hash slots, sharding, and failover", "databases"),
    ("Redis sorted sets for leaderboards and rate limiting", "databases"),
    ("Redis distributed locks with SET NX PX (Redlock algorithm)", "databases"),
    ("Redis pipeline and MULTI/EXEC transactions", "databases"),
    ("Redis cache eviction policies: LRU, LFU, volatile-ttl", "databases"),

    # ── Databases — MongoDB ───────────────────────────────────────────────────
    ("MongoDB document model: embedding vs referencing", "databases"),
    ("MongoDB aggregation pipeline: $match, $group, $lookup, $project", "databases"),
    ("MongoDB indexes: compound, sparse, TTL, text, geospatial", "databases"),
    ("MongoDB transactions and multi-document ACID guarantees", "databases"),
    ("MongoDB replica sets: primary, secondary, arbiter", "databases"),
    ("MongoDB sharding: shard key selection and chunk balancing", "databases"),
    ("MongoDB change streams for real-time event processing", "databases"),
    ("MongoDB schema design patterns: bucket, outlier, attribute", "databases"),

    # ── Databases — Cassandra ─────────────────────────────────────────────────
    ("Cassandra data modeling: partition key and clustering key", "databases"),
    ("Cassandra consistency levels: ONE, QUORUM, ALL", "databases"),
    ("Cassandra compaction strategies: STCS, LCS, TWCS", "databases"),
    ("Cassandra wide row pattern for time-series data", "databases"),
    ("Cassandra anti-patterns: joins, secondary indexes, large partitions", "databases"),
    ("Cassandra CQL vs SQL: limitations and workarounds", "databases"),

    # ── Databases — ClickHouse & Analytics ────────────────────────────────────
    ("ClickHouse MergeTree engine and partition pruning", "databases"),
    ("ClickHouse materialized views for pre-aggregation", "databases"),
    ("ClickHouse ReplicatedMergeTree for high availability", "databases"),
    ("ClickHouse columnar storage and compression for analytics", "databases"),
    ("Time-series data modeling in InfluxDB and TimescaleDB", "databases"),
    ("Data warehouse schema design: star vs snowflake", "databases"),
    ("OLAP cube design for multi-dimensional analysis", "databases"),

    # ── Databases — General patterns ──────────────────────────────────────────
    ("Index selectivity and query optimization decisions", "databases"),
    ("Composite index vs covering index vs partial index", "databases"),
    ("N+1 query problem and eager loading solutions", "databases"),
    ("Connection pooling best practices", "databases"),
    ("Optimistic vs pessimistic locking strategies", "databases"),
    ("Upsert patterns: INSERT ON CONFLICT and MERGE", "databases"),
    ("Soft delete vs hard delete trade-offs", "databases"),
    ("Temporal tables and bi-temporal data modeling", "databases"),
    ("Database migration strategies: expand-contract pattern", "databases"),
    ("Idempotent database operations for safe retries", "databases"),
    ("Read replica routing for query load distribution", "databases"),
    ("Database circuit breaker pattern", "databases"),
    ("Polyglot persistence: choosing the right database per use case", "databases"),
    ("CQRS with separate read and write datastores", "databases"),

    # ── Networking — foundational ─────────────────────────────────────────────
    ("TCP three-way handshake and connection termination (FIN/FIN-ACK)", "networking"),
    ("TCP flow control: sliding window and receiver buffer", "networking"),
    ("TCP congestion control: slow start, AIMD, BBR, CUBIC", "networking"),
    ("UDP characteristics and when to prefer UDP over TCP", "networking"),
    ("IP fragmentation and MTU path discovery", "networking"),
    ("IPv4 vs IPv6: addressing, headers, and transition mechanisms", "networking"),
    ("CIDR notation and subnetting: /24, /16, /8 explained", "networking"),
    ("NAT: source NAT, destination NAT, masquerade", "networking"),
    ("ARP: address resolution and ARP spoofing attacks", "networking"),
    ("ICMP: ping, traceroute, and path MTU discovery", "networking"),

    # ── Networking — DNS ──────────────────────────────────────────────────────
    ("DNS resolution: recursive, iterative, and authoritative queries", "networking"),
    ("DNS record types: A, AAAA, CNAME, MX, TXT, NS, SOA, SRV", "networking"),
    ("DNS TTL and cache behavior across resolvers", "networking"),
    ("DNSSEC: chain of trust and zone signing", "networking"),
    ("DNS-over-HTTPS and DNS-over-TLS for privacy", "networking"),
    ("Split-horizon DNS for internal vs external resolution", "networking"),
    ("DNS load balancing and GeoDNS for traffic routing", "networking"),

    # ── Networking — HTTP & Web ───────────────────────────────────────────────
    ("HTTP/1.1 persistent connections and head-of-line blocking", "networking"),
    ("HTTP/2 multiplexing, header compression (HPACK), and push", "networking"),
    ("HTTP/3 and QUIC: 0-RTT, connection migration, UDP basis", "networking"),
    ("TLS 1.3 handshake: key exchange, cipher suites, 0-RTT resumption", "networking"),
    ("TLS certificate chain: root CA, intermediate CA, leaf certificate", "networking"),
    ("HTTP caching: Cache-Control, ETag, Last-Modified, Vary", "networking"),
    ("WebSocket protocol: upgrade handshake, framing, heartbeats", "networking"),
    ("WebRTC: ICE, STUN, TURN, SDP offer/answer exchange", "networking"),
    ("Server-sent events vs WebSocket vs long polling", "networking"),
    ("HTTP/2 server push and its deprecation in practice", "networking"),
    ("CORS: preflight, Access-Control headers, and SameSite cookies", "networking"),
    ("gRPC over HTTP/2: protobuf, streaming, metadata", "networking"),
    ("MQTT: publish-subscribe, QoS levels 0/1/2, retained messages", "networking"),

    # ── Networking — routing & infrastructure ─────────────────────────────────
    ("BGP routing: AS paths, route selection, and prefix hijacking", "networking"),
    ("OSPF link-state routing and Dijkstra in network routers", "networking"),
    ("ECMP load balancing across multiple paths", "networking"),
    ("Anycast routing for DDoS mitigation and CDN edge selection", "networking"),
    ("VPN tunneling: IPsec, WireGuard, OpenVPN compared", "networking"),
    ("VLAN segmentation and 802.1Q tagging", "networking"),
    ("Software-defined networking: control plane vs data plane", "networking"),
    ("eBPF for network observability and packet filtering", "networking"),
    ("Network address translation: NAT traversal for P2P", "networking"),
    ("TCP BBR congestion control for high-bandwidth long-latency links", "networking"),

    # ── Networking — load balancing & proxying ────────────────────────────────
    ("Layer 4 vs Layer 7 load balancing trade-offs", "networking"),
    ("Consistent hashing for load balancer session affinity", "networking"),
    ("Nginx as reverse proxy: upstream, proxy_pass, keepalive", "networking"),
    ("HAProxy ACLs and backend selection algorithms", "networking"),
    ("Service mesh sidecar proxy: Envoy traffic management", "networking"),
    ("CDN edge caching, cache invalidation, and origin shield", "networking"),
    ("Health checks: active vs passive, circuit breaker integration", "networking"),

    # ── Networking — security ─────────────────────────────────────────────────
    ("DDoS mitigation: rate limiting, anycast, scrubbing centers", "networking"),
    ("Firewall rules: stateful vs stateless packet inspection", "networking"),
    ("Network segmentation with DMZ and micro-segmentation", "networking"),
    ("Zero trust networking: identity-based access control", "networking"),
    ("mTLS mutual authentication for service-to-service security", "networking"),
    ("Network intrusion detection: Snort, Suricata signatures", "networking"),
    ("TLS pinning for mobile apps and its trade-offs", "networking"),
    ("Packet capture with tcpdump and Wireshark for debugging", "networking"),

    # ── Data structures (practical) ───────────────────────────────────────────
    ("When to use a deque vs list in Python", "data_structures"),
    ("Circular buffer implementation and use cases", "data_structures"),
    ("LRU cache with OrderedDict or doubly linked list", "data_structures"),
    ("Monotonic stack for next greater element problems", "data_structures"),
    ("Union-Find with path compression and union by rank", "data_structures"),
    ("Segment tree with lazy propagation", "data_structures"),
    ("Persistent data structures and immutability", "data_structures"),
    ("Rope data structure for string editing", "data_structures"),
    ("Interval tree for overlapping interval queries", "data_structures"),
    ("Adjacency list vs adjacency matrix for graphs", "data_structures"),
    ("Hash collision resolution: chaining vs open addressing", "data_structures"),
    ("B+ tree: why databases prefer it over binary trees", "data_structures"),
    ("LSM tree: used in LevelDB, Cassandra, RocksDB", "data_structures"),

    # ── Algorithms ────────────────────────────────────────────────────────────
    ("Sliding window technique for subarray problems", "algorithms"),
    ("Two pointers technique for sorted array problems", "algorithms"),
    ("Fast and slow pointers for cycle detection", "algorithms"),
    ("BFS for shortest path in unweighted graphs", "algorithms"),
    ("DFS for topological sort and cycle detection", "algorithms"),
    ("Knapsack 0/1 dynamic programming", "algorithms"),
    ("Longest common subsequence dynamic programming", "algorithms"),
    ("Edit distance (Levenshtein) DP", "algorithms"),
    ("KMP string matching algorithm", "algorithms"),
    ("Rabin-Karp rolling hash for pattern matching", "algorithms"),
    ("Boyer-Moore voting for majority element", "algorithms"),
    ("Reservoir sampling for streaming data", "algorithms"),
    ("Counting sort and radix sort for linear time", "algorithms"),
    ("Sieve of Eratosthenes for prime generation", "algorithms"),
    ("Modular exponentiation for large powers", "algorithms"),

    # ── Concurrency ───────────────────────────────────────────────────────────
    ("Race condition prevention with locks and atomics", "concurrency"),
    ("Deadlock prevention: lock ordering and timeout", "concurrency"),
    ("Producer-consumer with bounded buffer", "concurrency"),
    ("Thread pool executor for CPU-bound tasks", "concurrency"),
    ("Asyncio coroutines for I/O-bound concurrency", "concurrency"),
    ("Lock-free programming with compare-and-swap", "concurrency"),
    ("Read-write lock for concurrent reads", "concurrency"),
    ("Actor model for message-passing concurrency", "concurrency"),

    # ── Architecture ──────────────────────────────────────────────────────────
    ("REST API versioning strategies", "architecture"),
    ("Idempotent HTTP operations and retry safety", "architecture"),
    ("Pagination: cursor-based vs offset-based", "architecture"),
    ("API rate limiting: token bucket and sliding window", "architecture"),
    ("gRPC vs REST vs GraphQL trade-offs", "architecture"),
    ("Webhook design and delivery guarantees", "architecture"),
    ("Circuit breaker pattern for fault tolerance", "architecture"),
    ("Saga pattern for distributed transactions", "architecture"),
    ("Outbox pattern for reliable message publishing", "architecture"),
    ("CQRS: separating read and write models", "architecture"),
    ("Event sourcing: state from events", "architecture"),
    ("Consistent hashing for load distribution", "architecture"),
    ("Two-phase commit vs saga for distributed transactions", "architecture"),
    ("Chaos engineering principles and practices", "architecture"),

    # ── Security ──────────────────────────────────────────────────────────────
    ("OWASP Top 10: injection flaws and prevention", "cybersecurity"),
    ("SQL injection prevention with parameterized queries", "cybersecurity"),
    ("XSS prevention: output encoding and CSP headers", "cybersecurity"),
    ("CSRF prevention with SameSite cookies and tokens", "cybersecurity"),
    ("Password hashing with bcrypt vs argon2", "cybersecurity"),
    ("JWT validation: signature, expiry, and claims", "cybersecurity"),
    ("OAuth2 authorization code flow with PKCE", "cybersecurity"),
    ("Secrets management: vault, environment variables, KMS", "cybersecurity"),
    ("TLS mutual authentication (mTLS)", "cybersecurity"),
    ("Smart contract security: reentrancy, overflow, access control", "cybersecurity"),

    # ── Testing ───────────────────────────────────────────────────────────────
    ("Test pyramid: unit, integration, e2e ratio", "testing"),
    ("Property-based testing with Hypothesis", "testing"),
    ("Mocking external dependencies for isolation", "testing"),
    ("Contract testing with Pact for microservices", "testing"),
    ("Performance testing: load, stress, soak tests", "testing"),
    ("Test double patterns: stub, spy, mock, fake", "testing"),

    # ── Performance ───────────────────────────────────────────────────────────
    ("CPU cache optimization: data locality and struct layout", "performance"),
    ("SIMD instructions for data-parallel computation", "performance"),
    ("Branch prediction and avoiding mispredictions", "performance"),
    ("I/O patterns: buffered, direct, memory-mapped", "performance"),
    ("Profiling Python with cProfile and py-spy", "performance"),
    ("HTTP/2 vs HTTP/3 performance characteristics", "performance"),
    ("Database query explain plan interpretation", "performance"),
    ("Caching strategies: write-through, write-back, write-around", "performance"),
    ("CDN edge caching and cache invalidation", "performance"),

    # ── Best practices ────────────────────────────────────────────────────────
    ("SOLID principles applied in practice", "best_practices"),
    ("Composition over inheritance in practice", "best_practices"),
    ("Feature flags for safe deployments", "best_practices"),
    ("Semantic versioning for library releases", "best_practices"),
    ("Structured logging with correlation IDs", "best_practices"),
    ("Observability: metrics, logs, and traces (OpenTelemetry)", "best_practices"),
    ("12-factor app methodology for cloud-native applications", "best_practices"),
    ("ADRs (Architecture Decision Records) for team alignment", "best_practices"),
]


# ─── EU generation prompt (language-aware) ───────────────────────────────────

EU_GENERATION_PROMPT = """You are a structured software engineering knowledge base.

Generate a precise, accurate EpistemicUnit for this concept:
"{concept}"

Respond in EXACTLY this JSON format (no other text):
{{
  "proposition": "One clear sentence describing what this IS or DOES (30-100 words)",
  "time_complexity": "Big-O time complexity if applicable, else null",
  "space_complexity": "Big-O space complexity if applicable, else null",
  "when_to_use": "2-3 sentence practical guidance on when to apply this (null if not applicable)",
  "tradeoffs": "Key advantages and disadvantages, briefly (null if not applicable)",
  "code_snippet": "Concise code example (10-40 lines) in the most relevant language demonstrating the concept, else null",
  "confidence": 0.85
}}"""


def call_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 120) -> str | None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 600},
        }).encode()
        req = urllib.request.Request(
            f"{host.rstrip('/')}/api/generate",
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
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def check_ollama_running() -> bool:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _model_exists(name: str) -> bool:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return any(
                m["name"] == name or m["name"].startswith(name + ":")
                for m in data.get("models", [])
            )
    except Exception:
        return False


EPICHAT_DATA = os.path.join(EPICHAT_DIR, "episteme_data")


def main():
    # Unbuffered output for real-time progress when run via subprocess (e.g. Kaggle)
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

    kg = KnowledgeGraph()
    kg.load(EPICHAT_DATA)
    start_count = len(kg.units)
    print(f"Starting EU count: {start_count}", flush=True)
    print(f"Target: 15,000+ EUs\n", flush=True)

    total_added = 0

    # ── Phase 1: Wikipedia scraping ───────────────────────────────────────────
    print(f"=== Phase 1: Wikipedia scraping ({len(WIKI_TOPICS)} topics) ===", flush=True)
    seeder = CodeSeeder(kg)
    n = seeder.seed_wikipedia(topics=WIKI_TOPICS, max_sentences=25)
    total_added += n
    print(f"  Added {n} EUs from Wikipedia. Total: {len(kg.units)}\n", flush=True)
    print("  [Phase 1] Saving knowledge graph (units.json, faiss.index)...", flush=True)
    kg.save(EPICHAT_DATA)
    print("  [Phase 1] Save complete.\n", flush=True)

    # ── Phase 2: LLM synthesis ────────────────────────────────────────────────
    print(f"=== Phase 2: LLM synthesis ({len(LLM_CONCEPTS)} concepts) ===", flush=True)

    ollama_ok = check_ollama_running()
    if not ollama_ok:
        print("  [WARN] Ollama not running. Trying to start...", flush=True)
        ollama_log = os.environ.get("TMP_DIR", "/tmp") + "/ollama.log"
        os.system(f"ollama serve > {ollama_log} 2>&1 &")
        time.sleep(5)
        ollama_ok = check_ollama_running()

    if not ollama_ok:
        print("  [SKIP] Ollama not available — skipping LLM synthesis", flush=True)
    else:
        # Pick best available model (prefer deepseek-r1 for quality)
        for candidate in ("deepseek-r1:latest", "llama3.1:8b"):
            if _model_exists(candidate):
                model = candidate
                break
        else:
            model = "llama3.1:8b"
        print(f"  Using model: {model}", flush=True)
        # Brief wait for Ollama to finish loading model on first request (Kaggle)
        print("  [INFO] First Ollama call may take 30-60s while model loads...", flush=True)

        source = Source(name=f"LLM-synthesized ({model})", reliability_score=0.82)
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
                confidence=float(parsed.get("confidence", 0.82)),
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
                print(f"  [{i+1:03d}/{len(LLM_CONCEPTS)}] +1 | {concept[:65]}", flush=True)
            else:
                n_skip += 1
                print(f"  [{i+1:03d}/{len(LLM_CONCEPTS)}] dup | {concept[:65]}", flush=True)

            if n_llm > 0 and n_llm % 50 == 0:
                kg.save(EPICHAT_DATA)
                print(f"  [SAVE] {len(kg.units)} total EUs", flush=True)

        print(f"\n  LLM synthesis: {n_llm} added, {n_skip} skipped", flush=True)

    # ── Final save & re-export traces ─────────────────────────────────────────
    print(f"\n=== Final save ===", flush=True)
    kg.save(EPICHAT_DATA)
    print(f"Total EU count: {len(kg.units)} (added {total_added} in this run, started at {start_count})", flush=True)

    print("\n=== Re-exporting training traces ===", flush=True)
    import subprocess
    result = subprocess.run(
        [sys.executable, "-u", "-m", "train.epichat_export",
         "--epichat-dir", EPICHAT_DIR,
         "--output", "data/epichat_traces.jsonl",
         "--min-confidence", "0.4"],
        cwd=os.environ.get("REPO_ROOT", str(_repo)),
        env={**os.environ, "EPICHAT_DIR": EPICHAT_DIR, "REPO_ROOT": os.environ.get("REPO_ROOT", str(_repo))},
        capture_output=True,
        text=True,
    )
    print(f"Trace export exit code: {result.returncode}", flush=True)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
