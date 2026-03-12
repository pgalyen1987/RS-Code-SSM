"""
Foundational programming axioms, best practices, algorithms, and design patterns.
Each entry is a tuple of:
  (proposition, domain, confidence, knowledge_type, extras_dict)

extras_dict keys (all optional):
  code_snippet, language, time_complexity, space_complexity, when_to_use, tradeoffs, keywords
"""

from epichat.core.epistemic_unit import KnowledgeType

# fmt: off
CODE_KNOWLEDGE = [

    # ------------------------------------------------------------------ #
    # PROGRAMMING AXIOMS — universal truths of software engineering       #
    # ------------------------------------------------------------------ #

    ("Readable code is more valuable than clever code; humans read code far more than they write it.",
     "best_practices", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Always", "keywords": ["readability", "maintainability", "clarity"]}),

    ("DRY (Don't Repeat Yourself): every piece of knowledge must have a single, unambiguous representation in a system.",
     "best_practices", 0.96, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "When the same logic appears in more than one place",
      "tradeoffs": "Over-abstraction can reduce readability; apply judgment",
      "keywords": ["DRY", "duplication", "abstraction"]}),

    ("YAGNI (You Aren't Gonna Need It): do not add functionality until it is necessary.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "During design and implementation",
      "tradeoffs": "Can conflict with designing for extensibility",
      "keywords": ["YAGNI", "over-engineering", "simplicity"]}),

    ("KISS (Keep It Simple, Stupid): simplicity should be a key goal and unnecessary complexity should be avoided.",
     "best_practices", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["KISS", "simplicity", "complexity"]}),

    ("SOLID: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Object-oriented design",
      "keywords": ["SOLID", "OOP", "design principles"]}),

    ("Single Responsibility Principle: a class should have only one reason to change.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "When designing classes or modules",
      "tradeoffs": "Can lead to many small classes",
      "keywords": ["SRP", "SOLID", "cohesion"]}),

    ("Open/Closed Principle: software entities should be open for extension but closed for modification.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Requires upfront abstraction; may over-engineer simple cases",
      "keywords": ["OCP", "SOLID", "extension"]}),

    ("Liskov Substitution Principle: objects of a subclass must be replaceable with objects of the superclass without altering correctness.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["LSP", "SOLID", "inheritance", "polymorphism"]}),

    ("Dependency Inversion Principle: depend on abstractions, not concretions.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Adds indirection; increases initial complexity",
      "keywords": ["DIP", "SOLID", "dependency injection"]}),

    ("Premature optimization is the root of all evil (Knuth): profile first, then optimize.",
     "best_practices", 0.96, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "After correctness is confirmed and profiling reveals bottleneck",
      "keywords": ["optimization", "profiling", "performance"]}),

    ("Code that is not tested is broken by definition; write tests before assuming code is correct.",
     "testing", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["testing", "TDD", "correctness"]}),

    ("Fail fast: detect and report errors as early as possible rather than continuing in an invalid state.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["error handling", "validation", "defensive programming"]}),

    ("Separation of Concerns: divide a program into distinct features with minimal overlap.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["SoC", "modularity", "architecture"]}),

    ("The Law of Demeter: a module should not know about the innards of the objects it manipulates.",
     "best_practices", 0.90, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Can increase number of wrapper methods",
      "keywords": ["Law of Demeter", "coupling", "encapsulation"]}),

    ("Composition over inheritance: favor object composition over class inheritance for flexibility.",
     "best_practices", 0.92, KnowledgeType.BEST_PRACTICE,
     {"tradeoffs": "Can result in more boilerplate code",
      "keywords": ["composition", "inheritance", "OOP", "flexibility"]}),

    ("Make it work, make it right, make it fast — in that order.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["correctness", "refactoring", "optimization"]}),

    ("Code review reduces defects: peer review catches bugs that authors miss due to cognitive bias.",
     "best_practices", 0.91, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["code review", "quality", "collaboration"]}),

    ("Version control every project: code without version control is one mistake away from disaster.",
     "best_practices", 0.98, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["git", "version control", "VCS"]}),

    ("Naming is the most important act in programming: good names eliminate the need for comments.",
     "best_practices", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["naming", "variables", "functions", "readability"]}),

    ("Functions should do one thing and do it well; if a function does more than one thing, split it.",
     "best_practices", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["functions", "SRP", "decomposition"]}),

    ("Magic numbers are a code smell: replace literals with named constants.",
     "best_practices", 0.92, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["magic numbers", "constants", "readability"]}),

    ("Always handle errors explicitly; swallowing exceptions silently creates ghost bugs.",
     "best_practices", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["error handling", "exceptions", "robustness"]}),

    ("Write code for the next programmer (which may be you in 6 months), not for the computer.",
     "best_practices", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["documentation", "readability", "maintainability"]}),

    # ------------------------------------------------------------------ #
    # DATA STRUCTURES                                                      #
    # ------------------------------------------------------------------ #

    ("Array: contiguous block of memory storing elements of the same type; O(1) random access by index.",
     "data_structures", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "Access O(1), Search O(n), Insert O(n), Delete O(n)",
      "space_complexity": "O(n)",
      "when_to_use": "When you need fast random access and the size is known",
      "tradeoffs": "Insertion/deletion are expensive; fixed size in some languages",
      "keywords": ["array", "list", "random access", "contiguous"]}),

    ("Linked List: sequence of nodes each holding data and a pointer to the next node; O(1) insert/delete at known position.",
     "data_structures", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "Access O(n), Search O(n), Insert O(1), Delete O(1)",
      "space_complexity": "O(n)",
      "when_to_use": "Frequent insertions/deletions; size unknown at start",
      "tradeoffs": "No random access; extra memory for pointers; poor cache locality",
      "keywords": ["linked list", "pointer", "node", "dynamic"]}),

    ("Hash Map (Hash Table): key-value store using a hash function for O(1) average-case lookup.",
     "data_structures", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "Search/Insert/Delete: O(1) average, O(n) worst case",
      "space_complexity": "O(n)",
      "when_to_use": "Fast lookup by key; counting frequencies; memoization",
      "tradeoffs": "Unordered; hash collisions; resizing cost; memory overhead",
      "keywords": ["hash map", "dictionary", "hash table", "O(1)", "lookup"]}),

    ("Stack (LIFO): last-in first-out structure supporting push/pop in O(1); used for call stacks, undo, parsing.",
     "data_structures", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "Push/Pop/Peek: O(1)",
      "space_complexity": "O(n)",
      "when_to_use": "Expression evaluation, DFS, undo/redo, balanced parentheses",
      "keywords": ["stack", "LIFO", "push", "pop", "DFS"]}),

    ("Queue (FIFO): first-in first-out structure; use deque for O(1) enqueue and dequeue.",
     "data_structures", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "Enqueue/Dequeue: O(1)",
      "space_complexity": "O(n)",
      "when_to_use": "BFS, task scheduling, producer-consumer, buffering",
      "tradeoffs": "Not suitable for random access",
      "keywords": ["queue", "FIFO", "BFS", "deque", "scheduling"]}),

    ("Binary Search Tree (BST): sorted binary tree; O(log n) average-case search, insert, delete.",
     "data_structures", 0.98, KnowledgeType.ALGORITHM,
     {"time_complexity": "Search/Insert/Delete: O(log n) average, O(n) worst",
      "space_complexity": "O(n)",
      "when_to_use": "Ordered data with frequent search, insertion, deletion",
      "tradeoffs": "Can degenerate to O(n) without balancing; use AVL/Red-Black for guarantees",
      "keywords": ["BST", "binary search tree", "sorted", "tree"]}),

    ("Heap (Priority Queue): complete binary tree satisfying heap property; O(log n) insert, O(1) peek min/max.",
     "data_structures", 0.98, KnowledgeType.ALGORITHM,
     {"time_complexity": "Insert O(log n), Extract-min/max O(log n), Peek O(1)",
      "space_complexity": "O(n)",
      "when_to_use": "Priority scheduling, Dijkstra, heap sort, top-K problems",
      "tradeoffs": "Not suitable for arbitrary search",
      "keywords": ["heap", "priority queue", "min-heap", "max-heap", "heapq"]}),

    ("Graph: set of nodes (vertices) and edges; can be directed/undirected, weighted/unweighted.",
     "data_structures", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Networks, social graphs, dependencies, maps, state machines",
      "keywords": ["graph", "vertex", "edge", "adjacency list", "adjacency matrix"]}),

    ("Trie (Prefix Tree): tree storing strings character by character; O(m) search where m = string length.",
     "data_structures", 0.97, KnowledgeType.ALGORITHM,
     {"time_complexity": "Insert/Search: O(m) where m = key length",
      "space_complexity": "O(n*m) worst case",
      "when_to_use": "Autocomplete, spell checking, prefix search, IP routing tables",
      "tradeoffs": "High memory usage; slower than hash map for exact lookup",
      "keywords": ["trie", "prefix tree", "autocomplete", "string"]}),

    # ------------------------------------------------------------------ #
    # ALGORITHMS                                                           #
    # ------------------------------------------------------------------ #

    ("Binary Search: find an element in a sorted array by repeatedly halving the search space; O(log n).",
     "algorithms", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(log n)",
      "space_complexity": "O(1) iterative, O(log n) recursive",
      "when_to_use": "Searching in sorted arrays; finding insertion points",
      "tradeoffs": "Requires sorted input; not suitable for linked lists",
      "code_snippet": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1""",
      "language": "Python",
      "keywords": ["binary search", "sorted", "O(log n)", "divide and conquer"]}),

    ("Merge Sort: stable, divide-and-conquer sort; guaranteed O(n log n) in all cases.",
     "algorithms", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n log n) best/average/worst",
      "space_complexity": "O(n)",
      "when_to_use": "When stability matters; sorting linked lists; external sorting",
      "tradeoffs": "Requires O(n) extra space; not in-place",
      "code_snippet": """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]""",
      "language": "Python",
      "keywords": ["merge sort", "stable sort", "divide and conquer", "O(n log n)"]}),

    ("Quick Sort: in-place divide-and-conquer sort; O(n log n) average, O(n²) worst case.",
     "algorithms", 0.98, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n log n) average, O(n²) worst",
      "space_complexity": "O(log n) stack space",
      "when_to_use": "General-purpose in-place sorting; fastest in practice for random data",
      "tradeoffs": "Unstable; worst case O(n²) on sorted input without random pivot",
      "keywords": ["quicksort", "partition", "pivot", "in-place", "O(n log n)"]}),

    ("Depth-First Search (DFS): explore as far as possible down each branch before backtracking; uses a stack.",
     "algorithms", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(V + E) where V=vertices, E=edges",
      "space_complexity": "O(V) call stack",
      "when_to_use": "Cycle detection, topological sort, connected components, maze solving",
      "tradeoffs": "May not find shortest path; can overflow stack on deep graphs",
      "code_snippet": """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited""",
      "language": "Python",
      "keywords": ["DFS", "depth-first", "graph traversal", "recursion", "stack"]}),

    ("Breadth-First Search (BFS): explore all neighbors level by level; uses a queue; finds shortest path in unweighted graphs.",
     "algorithms", 0.99, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(V + E)",
      "space_complexity": "O(V)",
      "when_to_use": "Shortest path in unweighted graphs; level-order traversal; web crawling",
      "tradeoffs": "Higher memory usage than DFS; not suitable for weighted shortest paths",
      "code_snippet": """from collections import deque

def bfs(graph, start):
    visited = {start}
    queue   = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited""",
      "language": "Python",
      "keywords": ["BFS", "breadth-first", "shortest path", "queue", "level order"]}),

    ("Dijkstra's algorithm: finds shortest paths from a source to all vertices in a weighted graph with non-negative edges.",
     "algorithms", 0.98, KnowledgeType.ALGORITHM,
     {"time_complexity": "O((V + E) log V) with binary heap",
      "space_complexity": "O(V)",
      "when_to_use": "GPS routing, network shortest paths, game AI pathfinding",
      "tradeoffs": "Cannot handle negative edge weights (use Bellman-Ford instead)",
      "keywords": ["Dijkstra", "shortest path", "weighted graph", "priority queue"]}),

    ("Dynamic Programming (DP): break problem into overlapping subproblems; store results to avoid recomputation.",
     "algorithms", 0.98, KnowledgeType.ALGORITHM,
     {"when_to_use": "Problem has optimal substructure and overlapping subproblems (e.g. Fibonacci, knapsack, LCS)",
      "tradeoffs": "Can use significant memory; sometimes harder to reason about than recursion",
      "code_snippet": """# Fibonacci with memoization (top-down DP)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)""",
      "language": "Python",
      "keywords": ["dynamic programming", "memoization", "tabulation", "optimal substructure"]}),

    ("Two-pointer technique: use two indices moving toward or away from each other to solve array problems in O(n).",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n)",
      "space_complexity": "O(1)",
      "when_to_use": "Sorted array pair-sum, container with most water, palindrome check",
      "keywords": ["two pointers", "sliding window", "array", "O(n)"]}),

    ("Sliding Window: maintain a window of elements satisfying a condition; avoids nested loops.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"time_complexity": "O(n)",
      "space_complexity": "O(1) or O(k) for window contents",
      "when_to_use": "Maximum subarray sum, longest substring without repeat, anagram search",
      "keywords": ["sliding window", "substring", "subarray", "O(n)"]}),

    ("Backtracking: explore all possibilities by incrementally building candidates and abandoning invalid ones.",
     "algorithms", 0.97, KnowledgeType.ALGORITHM,
     {"when_to_use": "N-queens, Sudoku, permutations, combinations, constraint satisfaction",
      "tradeoffs": "Exponential worst case; pruning is essential for performance",
      "keywords": ["backtracking", "recursion", "pruning", "combinatorics"]}),

    ("Greedy algorithm: make the locally optimal choice at each step; works when greedy choice property holds.",
     "algorithms", 0.96, KnowledgeType.ALGORITHM,
     {"when_to_use": "Interval scheduling, Huffman coding, Kruskal's/Prim's MST, coin change (some cases)",
      "tradeoffs": "Does not always produce global optimum; requires proof of correctness",
      "keywords": ["greedy", "optimal", "interval scheduling", "Huffman"]}),

    # ------------------------------------------------------------------ #
    # DESIGN PATTERNS                                                      #
    # ------------------------------------------------------------------ #

    ("Singleton pattern: ensure a class has only one instance and provide a global access point to it.",
     "design_patterns", 0.92, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Database connections, loggers, configuration managers",
      "tradeoffs": "Global state makes testing hard; hidden dependencies; not thread-safe without care",
      "code_snippet": """class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance""",
      "language": "Python",
      "keywords": ["singleton", "creational", "global state", "instance"]}),

    ("Factory Method pattern: define an interface for creating an object but let subclasses decide which class to instantiate.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "When the exact type of object to create isn't known until runtime",
      "tradeoffs": "Can proliferate subclasses",
      "keywords": ["factory", "creational", "interface", "polymorphism"]}),

    ("Observer pattern: define a one-to-many dependency so when one object changes state, all dependents are notified.",
     "design_patterns", 0.94, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Event systems, GUI callbacks, reactive programming, pub/sub",
      "tradeoffs": "Unexpected updates; memory leaks if observers not deregistered",
      "code_snippet": """class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event, *args, **kwargs):
        for cb in self._listeners.get(event, []):
            cb(*args, **kwargs)""",
      "language": "Python",
      "keywords": ["observer", "event", "pub/sub", "reactive", "behavioral"]}),

    ("Strategy pattern: define a family of algorithms, encapsulate each one, and make them interchangeable.",
     "design_patterns", 0.94, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Sorting strategies, payment methods, compression algorithms — when behavior varies",
      "tradeoffs": "Clients must be aware of different strategies",
      "keywords": ["strategy", "behavioral", "algorithm", "encapsulation", "interface"]}),

    ("Decorator pattern: attach additional responsibilities to an object dynamically without altering its class.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Adding logging, caching, authentication as cross-cutting concerns",
      "tradeoffs": "Can result in many small objects; order of decorators matters",
      "code_snippet": """import functools

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Done {func.__name__}")
        return result
    return wrapper""",
      "language": "Python",
      "keywords": ["decorator", "wrapper", "cross-cutting", "structural", "Python"]}),

    ("Repository pattern: abstract the data layer; provide a collection-like interface for accessing domain objects.",
     "design_patterns", 0.90, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "When you want to decouple business logic from data access",
      "tradeoffs": "Extra abstraction layer; can become leaky if not designed carefully",
      "keywords": ["repository", "data access", "abstraction", "ORM", "testability"]}),

    ("Command pattern: encapsulate a request as an object, enabling undo/redo and queuing of operations.",
     "design_patterns", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Undo/redo, transactional operations, task queues, macro recording",
      "keywords": ["command", "undo", "redo", "queue", "behavioral"]}),

    # ------------------------------------------------------------------ #
    # COMPLEXITY & BIG-O                                                   #
    # ------------------------------------------------------------------ #

    ("O(1) — constant time: operation takes the same time regardless of input size.",
     "complexity", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Hash map lookup, stack push/pop, array index access",
      "keywords": ["O(1)", "constant time", "Big-O", "complexity"]}),

    ("O(log n) — logarithmic time: input size is halved at each step; very efficient for large inputs.",
     "complexity", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Binary search, balanced BST operations, heap operations",
      "keywords": ["O(log n)", "logarithmic", "binary search", "Big-O"]}),

    ("O(n) — linear time: time grows proportionally to input size.",
     "complexity", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Single-pass array scan, linear search, counting",
      "keywords": ["O(n)", "linear", "Big-O", "complexity"]}),

    ("O(n log n) — linearithmic time: typical of efficient comparison sorts (merge sort, heap sort, quick sort average).",
     "complexity", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Sorting; the best possible complexity for comparison-based sort",
      "keywords": ["O(n log n)", "sorting", "merge sort", "quicksort"]}),

    ("O(n²) — quadratic time: nested loops over the same collection; avoid for large inputs.",
     "complexity", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Bubble sort, insertion sort (small arrays), brute-force pair comparisons",
      "tradeoffs": "Acceptable for small n; catastrophic for large n",
      "keywords": ["O(n²)", "quadratic", "nested loops", "bubble sort"]}),

    ("O(2ⁿ) — exponential time: solution space doubles with each addition; only feasible for very small n.",
     "complexity", 0.99, KnowledgeType.ALGORITHM,
     {"when_to_use": "Subset enumeration, brute-force combinatorics — only acceptable for tiny inputs",
      "tradeoffs": "Completely infeasible for n > ~30",
      "keywords": ["O(2^n)", "exponential", "combinatorics", "NP"]}),

    # ------------------------------------------------------------------ #
    # CONCURRENCY                                                          #
    # ------------------------------------------------------------------ #

    ("Race condition: two threads access shared data concurrently and at least one modifies it; outcome is non-deterministic.",
     "concurrency", 0.98, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Recognize when shared mutable state exists",
      "tradeoffs": "Hard to reproduce; requires synchronization",
      "keywords": ["race condition", "thread safety", "concurrency", "mutex"]}),

    ("Deadlock: two or more threads wait indefinitely for resources held by each other.",
     "concurrency", 0.98, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Prevent by: consistent lock ordering, timeout, avoid nested locks",
      "keywords": ["deadlock", "mutex", "lock", "concurrency"]}),

    ("Prefer immutability: immutable data eliminates entire classes of concurrency bugs.",
     "concurrency", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["immutability", "thread safety", "functional", "concurrency"]}),

    # ------------------------------------------------------------------ #
    # TESTING                                                              #
    # ------------------------------------------------------------------ #

    ("Test-Driven Development (TDD): write a failing test first, then write the minimal code to pass it, then refactor.",
     "testing", 0.91, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Greenfield development; catching regressions; designing APIs through usage",
      "tradeoffs": "Slower initial velocity; tests become a maintenance burden if poorly designed",
      "keywords": ["TDD", "red-green-refactor", "unit test", "test first"]}),

    ("Unit test a single unit of behavior in isolation; mock all external dependencies.",
     "testing", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["unit test", "mock", "isolation", "pytest"]}),

    ("Integration test validates that multiple components work together correctly.",
     "testing", 0.94, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["integration test", "component", "end-to-end"]}),

    ("A test should have one reason to fail: test one behavior per test function.",
     "testing", 0.93, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["unit test", "single assertion", "SRP"]}),

    ("The test pyramid: many unit tests, fewer integration tests, very few end-to-end tests.",
     "testing", 0.92, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Structuring a test suite for speed and reliability",
      "keywords": ["test pyramid", "unit", "integration", "E2E"]}),

    # ------------------------------------------------------------------ #
    # SECURITY                                                             #
    # ------------------------------------------------------------------ #

    ("Never trust user input: validate and sanitize all data from external sources before use.",
     "security", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["input validation", "injection", "XSS", "SQL injection"]}),

    ("SQL injection is prevented by parameterized queries / prepared statements, never string concatenation.",
     "security", 0.99, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """# WRONG — never do this
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# CORRECT — parameterized query
cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))""",
      "language": "Python",
      "keywords": ["SQL injection", "parameterized query", "security", "ORM"]}),

    ("Store passwords using a slow adaptive hash: bcrypt, scrypt, or Argon2 — never MD5 or SHA-1.",
     "security", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["password hashing", "bcrypt", "Argon2", "security"]}),

    ("Principle of least privilege: code should request only the permissions it needs.",
     "security", 0.97, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["least privilege", "permissions", "security", "access control"]}),

    # ------------------------------------------------------------------ #
    # PYTHON-SPECIFIC                                                      #
    # ------------------------------------------------------------------ #

    ("Python list comprehension is faster and more Pythonic than equivalent for-loop append.",
     "python", 0.93, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """# Slower
squares = []
for x in range(10):
    squares.append(x * x)

# Faster and Pythonic
squares = [x * x for x in range(10)]""",
      "language": "Python",
      "keywords": ["list comprehension", "Python", "Pythonic", "performance"]}),

    ("Use generators for large sequences to avoid loading all data into memory at once.",
     "python", 0.94, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """# Generator expression — lazy, memory-efficient
total = sum(x * x for x in range(10_000_000))""",
      "language": "Python",
      "keywords": ["generator", "yield", "memory", "lazy", "Python"]}),

    ("Python's GIL prevents true parallel CPU-bound threading; use multiprocessing or asyncio for concurrency.",
     "python", 0.95, KnowledgeType.EMPIRICAL,
     {"tradeoffs": "asyncio is cooperative concurrency (I/O-bound); multiprocessing has IPC overhead",
      "keywords": ["GIL", "multiprocessing", "asyncio", "threading", "Python"]}),

    ("Use dataclasses or Pydantic for data containers instead of plain dicts to get type safety and auto-generated methods.",
     "python", 0.91, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float""",
      "language": "Python",
      "keywords": ["dataclass", "Pydantic", "type hints", "Python"]}),

    ("Context managers (with statement) guarantee resource cleanup even when exceptions occur.",
     "python", 0.96, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """with open("file.txt") as f:
    data = f.read()
# file is always closed here""",
      "language": "Python",
      "keywords": ["context manager", "with", "resource management", "Python"]}),

    # ------------------------------------------------------------------ #
    # ARCHITECTURE                                                         #
    # ------------------------------------------------------------------ #

    ("MVC (Model-View-Controller): separate data (Model), presentation (View), and logic (Controller).",
     "architecture", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Web applications, GUI applications",
      "keywords": ["MVC", "architecture", "separation of concerns"]}),

    ("REST API: stateless, resource-based HTTP API; use nouns for endpoints, verbs for HTTP methods.",
     "architecture", 0.93, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Client-server web APIs",
      "tradeoffs": "Statelessness can increase payload size; not ideal for real-time",
      "keywords": ["REST", "API", "HTTP", "stateless", "resource"]}),

    ("Microservices: decompose application into small, independently deployable services communicating over APIs.",
     "architecture", 0.88, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Large teams, independent scaling needs, polyglot persistence",
      "tradeoffs": "Network latency, distributed systems complexity, data consistency challenges",
      "keywords": ["microservices", "architecture", "distributed", "scalability"]}),

    ("Event-driven architecture: components communicate by emitting and reacting to events; decouples producers and consumers.",
     "architecture", 0.89, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "High-throughput systems, real-time pipelines, loose coupling requirements",
      "tradeoffs": "Eventual consistency; harder to trace flows; ordering challenges",
      "keywords": ["event-driven", "pub/sub", "Kafka", "async", "decoupling"]}),

    ("CQRS (Command Query Responsibility Segregation): separate read and write models for scalability.",
     "architecture", 0.86, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "High-read/write ratio disparity; complex domain models",
      "tradeoffs": "Increased complexity; eventual consistency between read/write models",
      "keywords": ["CQRS", "command", "query", "scalability", "architecture"]}),

    # ------------------------------------------------------------------ #
    # TRADEOFFS                                                            #
    # ------------------------------------------------------------------ #

    ("Time-space tradeoff: you can often trade memory for speed (caching, memoization) or speed for memory (streaming).",
     "algorithms", 0.97, KnowledgeType.TRADEOFF,
     {"keywords": ["tradeoff", "cache", "memoization", "space", "time"]}),

    ("Abstraction has a cost: every layer of indirection adds cognitive load; only abstract when the benefit outweighs the cost.",
     "best_practices", 0.94, KnowledgeType.TRADEOFF,
     {"keywords": ["abstraction", "complexity", "indirection", "tradeoff"]}),

    ("Consistency vs availability (CAP theorem): a distributed system can guarantee at most two of: Consistency, Availability, Partition tolerance.",
     "distributed_systems", 0.97, KnowledgeType.TRADEOFF,
     {"keywords": ["CAP theorem", "consistency", "availability", "distributed", "partition"]}),

    ("Strong typing catches bugs at compile time; dynamic typing allows faster prototyping but moves errors to runtime.",
     "languages", 0.93, KnowledgeType.TRADEOFF,
     {"keywords": ["type system", "static", "dynamic", "TypeScript", "Python", "tradeoff"]}),

    # ------------------------------------------------------------------ #
    # OOP PRINCIPLES                                                       #
    # ------------------------------------------------------------------ #

    ("Encapsulation: bundle data and methods that operate on that data within a class; restrict direct access to internal state.",
     "oop", 0.98, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Always in OOP — expose behavior, hide state",
      "tradeoffs": "Getters/setters can be boilerplate; prefer meaningful methods over raw accessors",
      "keywords": ["encapsulation", "OOP", "private", "information hiding"]}),

    ("Inheritance: a class inherits state and behavior from a parent class, enabling code reuse and polymorphism.",
     "oop", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "True IS-A relationships; extending framework base classes",
      "tradeoffs": "Tight coupling; fragile base class problem; prefer composition for HAS-A",
      "keywords": ["inheritance", "OOP", "subclass", "superclass", "extends"]}),

    ("Polymorphism: one interface, multiple implementations; enables writing code that works with objects of different types.",
     "oop", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Plugin systems, strategy pattern, collections of related objects",
      "keywords": ["polymorphism", "OOP", "interface", "overriding", "duck typing"]}),

    ("Abstract class: cannot be instantiated directly; defines a contract for subclasses; can include partial implementation.",
     "oop", 0.96, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "Shared implementation exists but class itself is not a complete concept",
      "tradeoffs": "Locks subclasses into an inheritance hierarchy",
      "keywords": ["abstract class", "OOP", "template method", "inheritance"]}),

    ("Interface / Protocol: defines a contract (method signatures) with no implementation; enables loose coupling.",
     "oop", 0.97, KnowledgeType.BEST_PRACTICE,
     {"when_to_use": "When multiple unrelated classes should share a capability",
      "tradeoffs": "No default implementation (in classic interfaces); can cause interface explosion",
      "keywords": ["interface", "protocol", "contract", "duck typing", "OOP"]}),

    ("Method overriding vs overloading: overriding replaces parent behavior (runtime polymorphism); overloading defines same name with different params (compile-time).",
     "oop", 0.95, KnowledgeType.EMPIRICAL,
     {"keywords": ["overriding", "overloading", "polymorphism", "OOP"]}),

    ("Cohesion and coupling: aim for high cohesion (focused classes) and low coupling (minimal dependencies between classes).",
     "oop", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["cohesion", "coupling", "OOP", "design", "modularity"]}),

    # ------------------------------------------------------------------ #
    # MORE DESIGN PATTERNS                                                 #
    # ------------------------------------------------------------------ #

    ("Builder pattern: construct a complex object step by step; useful when constructor would have too many parameters.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Complex objects with many optional fields (e.g. query builders, HTTP request builders)",
      "tradeoffs": "More code; builder must be kept in sync with object",
      "code_snippet": """class QueryBuilder:
    def __init__(self):
        self._table = None
        self._conditions = []
        self._limit = None

    def from_table(self, table):
        self._table = table
        return self

    def where(self, condition):
        self._conditions.append(condition)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def build(self):
        sql = f"SELECT * FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql""",
      "language": "Python",
      "keywords": ["builder", "creational", "fluent interface", "complex object"]}),

    ("Adapter pattern: convert the interface of a class into another interface clients expect; makes incompatible classes work together.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Integrating legacy code; wrapping third-party libraries to match your domain interface",
      "keywords": ["adapter", "wrapper", "structural", "interface", "compatibility"]}),

    ("Facade pattern: provide a simplified interface to a complex subsystem.",
     "design_patterns", 0.93, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Hiding complexity of subsystems (e.g. a simple API over complex media codec libraries)",
      "tradeoffs": "Can become a god class if not managed",
      "keywords": ["facade", "structural", "simplification", "subsystem"]}),

    ("Proxy pattern: provide a surrogate for another object to control access, add caching, or add logging.",
     "design_patterns", 0.91, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Lazy initialization, access control, remote proxies, caching proxies",
      "keywords": ["proxy", "structural", "lazy", "access control", "cache"]}),

    ("Template Method pattern: define the skeleton of an algorithm in a base class; subclasses fill in the blanks.",
     "design_patterns", 0.92, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "When several classes share the same algorithm structure but differ in details",
      "tradeoffs": "Tightly couples subclasses to base class structure",
      "keywords": ["template method", "behavioral", "inheritance", "hook"]}),

    ("Iterator pattern: provide a standard way to traverse a collection without exposing its internal structure.",
     "design_patterns", 0.94, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Custom collections, lazy sequences, tree traversal",
      "code_snippet": """class CountUp:
    def __init__(self, limit):
        self._limit = limit
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current >= self._limit:
            raise StopIteration
        val = self._current
        self._current += 1
        return val""",
      "language": "Python",
      "keywords": ["iterator", "behavioral", "traversal", "__iter__", "__next__"]}),

    ("Dependency Injection: pass dependencies into a class rather than letting it create them; enables testability and flexibility.",
     "design_patterns", 0.95, KnowledgeType.DESIGN_PATTERN,
     {"when_to_use": "Unit testing with mocks; swapping implementations; frameworks (Flask, FastAPI, Spring)",
      "tradeoffs": "More wiring code; can be over-engineered for simple cases",
      "code_snippet": """class EmailService:
    def send(self, to, body): ...

class UserService:
    def __init__(self, email_service: EmailService):  # injected
        self._email = email_service

    def register(self, user):
        # ... create user ...
        self._email.send(user.email, "Welcome!")""",
      "language": "Python",
      "keywords": ["dependency injection", "DI", "IoC", "testability", "SOLID"]}),

    # ------------------------------------------------------------------ #
    # KOTLIN                                                               #
    # ------------------------------------------------------------------ #

    ("Kotlin is a statically typed JVM language designed to be fully interoperable with Java while eliminating common Java pitfalls.",
     "kotlin", 0.97, KnowledgeType.EMPIRICAL,
     {"keywords": ["Kotlin", "JVM", "Android", "Java interop", "null safety"]}),

    ("Kotlin null safety: the type system distinguishes nullable (String?) from non-null (String) types; eliminates NullPointerException at compile time.",
     "kotlin", 0.98, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """val name: String = "Alice"    // non-null, safe
val nick: String? = null     // nullable, must check

// Safe call — returns null if nick is null
val upper = nick?.uppercase()

// Elvis operator — provide default
val display = nick ?: "Anonymous"

// Not-null assertion — throws NPE if null (avoid)
val forced = nick!!.uppercase()""",
      "language": "Kotlin",
      "keywords": ["null safety", "Kotlin", "nullable", "elvis operator", "NPE"]}),

    ("Kotlin data classes automatically generate equals(), hashCode(), toString(), copy(), and component functions.",
     "kotlin", 0.97, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """data class User(val id: Int, val name: String, val email: String)

val alice = User(1, "Alice", "alice@example.com")
val bob   = alice.copy(id = 2, name = "Bob")
println(alice)  // User(id=1, name=Alice, email=alice@example.com)""",
      "language": "Kotlin",
      "keywords": ["data class", "Kotlin", "equals", "hashCode", "copy"]}),

    ("Kotlin extension functions add methods to existing classes without inheritance or modification.",
     "kotlin", 0.96, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """fun String.isPalindrome(): Boolean =
    this == this.reversed()

println("racecar".isPalindrome())  // true
println("hello".isPalindrome())    // false""",
      "language": "Kotlin",
      "keywords": ["extension function", "Kotlin", "fluent", "API design"]}),

    ("Kotlin coroutines provide lightweight concurrency; suspend functions pause without blocking threads.",
     "kotlin", 0.95, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """import kotlinx.coroutines.*

suspend fun fetchUser(id: Int): User = withContext(Dispatchers.IO) {
    api.getUser(id)  // non-blocking network call
}

fun main() = runBlocking {
    val user = fetchUser(42)
    println(user.name)
}""",
      "language": "Kotlin",
      "when_to_use": "Async I/O, parallel network calls, Android UI work",
      "tradeoffs": "Requires coroutine scope management; cancellation must be handled",
      "keywords": ["coroutines", "Kotlin", "suspend", "async", "concurrency"]}),

    ("Kotlin sealed classes model restricted class hierarchies; the compiler enforces exhaustive when expressions.",
     "kotlin", 0.96, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """sealed class Result<out T>
data class Success<T>(val value: T) : Result<T>()
data class Failure(val error: Exception) : Result<Nothing>()
object Loading : Result<Nothing>()

fun handle(r: Result<String>) = when (r) {
    is Success -> println(r.value)
    is Failure -> println(r.error.message)
    is Loading  -> println("Loading...")
    // No else needed — compiler ensures exhaustiveness
}""",
      "language": "Kotlin",
      "when_to_use": "Result types, state machines, AST nodes, algebraic data types",
      "keywords": ["sealed class", "Kotlin", "when", "exhaustive", "ADT"]}),

    ("Kotlin scope functions (let, run, with, apply, also) apply operations within an object's context.",
     "kotlin", 0.93, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """// apply — configure object, returns receiver
val user = User().apply {
    name = "Alice"
    email = "alice@example.com"
}

// let — transform nullable, returns lambda result
val length = name?.let { it.length } ?: 0""",
      "language": "Kotlin",
      "keywords": ["let", "apply", "run", "with", "also", "scope functions", "Kotlin"]}),

    ("Kotlin companion objects replace Java static members and can implement interfaces.",
     "kotlin", 0.94, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """class Config private constructor(val debug: Boolean) {
    companion object {
        fun fromEnv() = Config(debug = System.getenv("DEBUG") == "true")
    }
}

val cfg = Config.fromEnv()""",
      "language": "Kotlin",
      "keywords": ["companion object", "Kotlin", "static", "factory"]}),

    ("Kotlin smart casts: after a type-check the compiler automatically casts without an explicit cast.",
     "kotlin", 0.96, KnowledgeType.BEST_PRACTICE,
     {"code_snippet": """fun printLength(obj: Any) {
    if (obj is String) {
        println(obj.length)  // smart-cast to String — no explicit cast needed
    }
}""",
      "language": "Kotlin",
      "keywords": ["smart cast", "Kotlin", "is", "type check", "cast"]}),

    # ------------------------------------------------------------------ #
    # NETWORKING                                                           #
    # ------------------------------------------------------------------ #

    ("OSI model has 7 layers: Physical, Data Link, Network, Transport, Session, Presentation, Application.",
     "networking", 0.98, KnowledgeType.EMPIRICAL,
     {"keywords": ["OSI", "networking", "layers", "TCP/IP", "model"]}),

    ("TCP (Transmission Control Protocol): reliable, ordered, connection-oriented transport; uses three-way handshake.",
     "networking", 0.98, KnowledgeType.EMPIRICAL,
     {"when_to_use": "When data integrity and ordering matter: HTTP, SMTP, SSH, file transfer",
      "tradeoffs": "Higher latency than UDP due to acknowledgment overhead",
      "keywords": ["TCP", "reliable", "connection-oriented", "handshake", "networking"]}),

    ("UDP (User Datagram Protocol): connectionless, low-latency transport; no delivery guarantee.",
     "networking", 0.98, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Real-time applications where speed > reliability: DNS, VoIP, video streaming, games",
      "tradeoffs": "Packets may be lost, duplicated, or arrive out of order",
      "keywords": ["UDP", "connectionless", "low latency", "datagram", "networking"]}),

    ("HTTP/1.1 uses persistent connections; HTTP/2 adds multiplexing, header compression, server push; HTTP/3 uses QUIC over UDP.",
     "networking", 0.96, KnowledgeType.EMPIRICAL,
     {"keywords": ["HTTP", "HTTP/2", "HTTP/3", "QUIC", "multiplexing", "networking"]}),

    ("DNS (Domain Name System): translates human-readable domain names to IP addresses; hierarchical distributed database.",
     "networking", 0.98, KnowledgeType.EMPIRICAL,
     {"keywords": ["DNS", "domain", "IP", "resolver", "nameserver", "networking"]}),

    ("TLS (Transport Layer Security): cryptographic protocol providing authentication and encrypted communication over a network.",
     "networking", 0.98, KnowledgeType.EMPIRICAL,
     {"when_to_use": "All production web traffic (HTTPS), API calls, any sensitive data in transit",
      "keywords": ["TLS", "SSL", "HTTPS", "encryption", "certificate", "networking"]}),

    ("A socket is an endpoint for communication between two machines; identified by IP address + port number.",
     "networking", 0.98, KnowledgeType.EMPIRICAL,
     {"code_snippet": """import socket

# Simple TCP server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('localhost', 8080))
    s.listen()
    conn, addr = s.accept()
    with conn:
        data = conn.recv(1024)
        conn.sendall(data)  # echo back""",
      "language": "Python",
      "keywords": ["socket", "TCP", "server", "client", "networking"]}),

    ("Load balancer distributes incoming network traffic across multiple servers to ensure no single server is overloaded.",
     "networking", 0.96, KnowledgeType.EMPIRICAL,
     {"when_to_use": "High-availability, horizontal scaling",
      "keywords": ["load balancer", "nginx", "reverse proxy", "scalability", "networking"]}),

    ("CDN (Content Delivery Network): geographically distributed servers cache static assets close to users to reduce latency.",
     "networking", 0.96, KnowledgeType.EMPIRICAL,
     {"keywords": ["CDN", "cache", "latency", "static assets", "networking"]}),

    ("REST uses HTTP verbs: GET (read), POST (create), PUT/PATCH (update), DELETE (remove); responses use HTTP status codes.",
     "networking", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["REST", "HTTP", "GET", "POST", "PUT", "DELETE", "status codes"]}),

    ("WebSockets provide full-duplex communication over a single TCP connection; ideal for real-time applications.",
     "networking", 0.95, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Chat, live dashboards, collaborative editing, gaming",
      "tradeoffs": "More complex than HTTP; connection state must be managed",
      "keywords": ["WebSocket", "real-time", "full-duplex", "networking"]}),

    # ------------------------------------------------------------------ #
    # CYBERSECURITY                                                        #
    # ------------------------------------------------------------------ #

    ("Defense in depth: apply multiple layers of security controls so that if one fails, others still protect the system.",
     "cybersecurity", 0.98, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["defense in depth", "security layers", "cybersecurity"]}),

    ("OWASP Top 10 lists the most critical web application security risks: Injection, Broken Auth, XSS, IDOR, Security Misconfiguration, etc.",
     "cybersecurity", 0.97, KnowledgeType.EMPIRICAL,
     {"keywords": ["OWASP", "web security", "injection", "XSS", "IDOR", "cybersecurity"]}),

    ("Cross-Site Scripting (XSS): attacker injects malicious scripts into content that other users execute in their browser.",
     "cybersecurity", 0.98, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Prevent by escaping all user-supplied output; use Content Security Policy (CSP)",
      "keywords": ["XSS", "injection", "web security", "CSP", "sanitization"]}),

    ("Cross-Site Request Forgery (CSRF): tricks authenticated user's browser into submitting requests the user didn't intend.",
     "cybersecurity", 0.97, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Prevent with CSRF tokens, SameSite cookies, checking Origin/Referer headers",
      "keywords": ["CSRF", "web security", "session", "token"]}),

    ("Insecure Direct Object Reference (IDOR): exposing internal IDs in URLs/params lets attackers access other users' data.",
     "cybersecurity", 0.97, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Always authorize every request against the authenticated user's permissions",
      "keywords": ["IDOR", "authorization", "access control", "web security"]}),

    ("Authentication proves identity (who you are); Authorization determines access (what you can do); confusing them is a critical bug.",
     "cybersecurity", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["authentication", "authorization", "authn", "authz", "security"]}),

    ("JWT (JSON Web Token): compact, self-contained token for transmitting claims; must be verified using a secret or public key.",
     "cybersecurity", 0.95, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Stateless authentication; API auth between services",
      "tradeoffs": "Cannot be invalidated before expiry without a blocklist; payload is only base64 encoded, not encrypted",
      "keywords": ["JWT", "token", "authentication", "OAuth", "security"]}),

    ("Never log sensitive data: passwords, credit card numbers, tokens, PII should never appear in log files.",
     "cybersecurity", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["logging", "PII", "sensitive data", "security", "GDPR"]}),

    ("Cryptography: use established libraries (OpenSSL, libsodium, Python's cryptography); never implement your own crypto primitives.",
     "cybersecurity", 0.99, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["cryptography", "AES", "RSA", "libsodium", "security"]}),

    ("Rate limiting prevents brute-force attacks and denial-of-service by restricting the number of requests per time window.",
     "cybersecurity", 0.96, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["rate limiting", "brute force", "DoS", "API security"]}),

    ("Symmetric encryption (AES) uses the same key for encryption and decryption; fast, suitable for bulk data.",
     "cybersecurity", 0.97, KnowledgeType.EMPIRICAL,
     {"tradeoffs": "Key distribution problem — must securely share the key",
      "keywords": ["AES", "symmetric", "encryption", "cryptography"]}),

    ("Asymmetric encryption (RSA, ECDSA) uses a public/private key pair; used for key exchange and digital signatures.",
     "cybersecurity", 0.97, KnowledgeType.EMPIRICAL,
     {"when_to_use": "TLS key exchange, digital signatures, SSH, certificate verification",
      "tradeoffs": "Much slower than symmetric; not suitable for bulk data",
      "keywords": ["RSA", "ECDSA", "public key", "private key", "asymmetric", "PKI"]}),

    ("SQL injection: attacker inserts SQL code into input fields to manipulate the database query.",
     "cybersecurity", 0.99, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Always use parameterized queries or ORMs; never concatenate user input into SQL",
      "keywords": ["SQL injection", "parameterized query", "ORM", "security", "OWASP"]}),

    ("Man-in-the-Middle (MitM) attack: attacker secretly intercepts and possibly alters communication between two parties.",
     "cybersecurity", 0.97, KnowledgeType.EMPIRICAL,
     {"when_to_use": "Prevented by TLS with certificate pinning or proper CA validation",
      "keywords": ["MitM", "TLS", "certificate", "interception", "security"]}),

    ("Penetration testing (pentesting): authorized simulated attack to identify security vulnerabilities before attackers do.",
     "cybersecurity", 0.95, KnowledgeType.EMPIRICAL,
     {"keywords": ["pentesting", "penetration testing", "vulnerability", "red team", "security"]}),

    ("Security headers harden web apps: Content-Security-Policy, X-Frame-Options, Strict-Transport-Security, X-Content-Type-Options.",
     "cybersecurity", 0.95, KnowledgeType.BEST_PRACTICE,
     {"keywords": ["CSP", "HSTS", "security headers", "HTTP", "web security"]}),

]
# fmt: on
