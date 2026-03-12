"""
Expand EpiChat's knowledge graph with additional Wikipedia topics.

Adds ~100 more high-quality CS/ML/systems topics to the existing 350 EUs.
Run from the SSM directory:
    source scripts/env.sh  # or set EPICHAT_DIR, REPO_ROOT
    python scripts/expand_epichat.py

Target: 1000+ EUs covering ML, systems, databases, advanced algorithms,
        compilers, concurrency, distributed systems, and more.
"""

import os
import sys

# epichat is in-repo at REPO_ROOT/epichat — add REPO_ROOT so "epichat" is importable
_repo = __import__("pathlib").Path(__file__).resolve().parent.parent
REPO_ROOT = os.environ.get("REPO_ROOT") or str(_repo)
EPICHAT_DIR = os.environ.get("EPICHAT_DIR") or str(_repo / "epichat")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from epichat.core.knowledge_graph import KnowledgeGraph
from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from epichat.seeding.code_seeder import CodeSeeder

# ─── Additional topics ────────────────────────────────────────────────────────

EXTRA_TOPICS = [
    # Machine Learning & AI
    "Machine learning",
    "Deep learning",
    "Neural network (machine learning)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Backpropagation",
    "Gradient descent",
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
    "Fine-tuning (deep learning)",
    "Batch normalization",
    "Dropout (neural networks)",

    # Databases
    "Database",
    "Relational database",
    "SQL",
    "NoSQL",
    "Database index",
    "B-tree",
    "Database normalization",
    "ACID",
    "Database transaction",
    "Query optimization",
    "PostgreSQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "CAP theorem",
    "Eventual consistency",

    # Distributed Systems
    "Distributed computing",
    "Consensus (computer science)",
    "Raft (algorithm)",
    "Paxos (computer science)",
    "MapReduce",
    "Apache Kafka",
    "Message queue",
    "Remote procedure call",
    "gRPC",
    "Distributed hash table",
    "Consistent hashing",
    "Sharding",
    "Replication (computing)",
    "Two-phase commit protocol",
    "Saga pattern",

    # Systems & OS
    "Operating system",
    "Process (computing)",
    "Thread (computing)",
    "Concurrency (computer science)",
    "Deadlock",
    "Semaphore (programming)",
    "Mutex",
    "Memory management",
    "Virtual memory",
    "Cache (computing)",
    "CPU cache",
    "Garbage collection (computer science)",
    "Compiler",
    "Just-in-time compilation",
    "Interpreter (computing)",
    "Abstract syntax tree",
    "Parsing",
    "Lexical analysis",
    "Docker (software)",
    "Kubernetes",
    "Continuous deployment",

    # Advanced Algorithms
    "Greedy algorithm",
    "Divide-and-conquer algorithm",
    "Backtracking",
    "Branch and bound",
    "Approximation algorithm",
    "Randomized algorithm",
    "Amortized analysis",
    "Fibonacci heap",
    "Disjoint-set data structure",
    "Segment tree",
    "Fenwick tree",
    "Trie",
    "Suffix array",
    "String-searching algorithm",
    "Bloom filter",
    "Count–min sketch",
    "Skip list",
    "Red–black tree",
    "AVL tree",
    "Topological sorting",
    "Strongly connected component",
    "Minimum spanning tree",
    "Dijkstra's algorithm",
    "A* search algorithm",
    "Floyd–Warshall algorithm",
    "Bellman–Ford algorithm",
    "Network flow",
    "Maximum flow problem",
    "Linear programming",
    "Integer programming",

    # Concurrency Patterns
    "Actor model",
    "Futures and promises",
    "Coroutine",
    "Event loop",
    "Asynchronous I/O",
    "Lock-free data structure",
    "Software transactional memory",

    # Software Engineering
    "Clean code",
    "Domain-driven design",
    "Test doubles",
    "Dependency inversion principle",
    "Interface segregation principle",
    "Open–closed principle",
    "Law of Demeter",
    "Coupling (computer programming)",
    "Cohesion (computer science)",
    "Code smell",
    "Anti-pattern",
    "Software metric",
    "Cyclomatic complexity",

    # Cloud & Infrastructure
    "Cloud computing",
    "Serverless computing",
    "Infrastructure as code",
    "Service mesh",
    "API gateway",
    "Circuit breaker (computing)",
    "Rate limiting",
    "Chaos engineering",
]


def main():
    kg = KnowledgeGraph(os.path.join(EPICHAT_DIR, "episteme_data"))
    print(f"Current EU count: {len(kg.units)}")

    seeder = CodeSeeder(kg)

    print(f"\nSeeding {len(EXTRA_TOPICS)} additional Wikipedia topics...")
    n = seeder.seed_wikipedia(
        topics=EXTRA_TOPICS,
        max_sentences=25,
    )
    print(f"\nAdded {n} new EUs")
    print(f"Total EU count: {len(kg.units)}")

    print("\nSaving...")
    kg.save()
    print("Done.")

    # Also re-export training traces
    print("\nRe-exporting training traces...")
    repo = os.environ.get("REPO_ROOT", str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    os.system(
        f"cd {repo} && (source .venv/bin/activate 2>/dev/null || true) && "
        f"python -u -m train.epichat_export "
        f"--epichat-dir {EPICHAT_DIR} "
        "--output data/epichat_traces.jsonl "
        "--min-confidence 0.4"
    )


if __name__ == "__main__":
    main()
