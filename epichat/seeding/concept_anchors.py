"""
Universal programming concept anchor EUs.

Each anchor is a language-agnostic EU that language-specific EUs
link back to via RelationType.SPECIALIZES. This lets the graph connect
"Python for loop", "Java enhanced for loop", "Kotlin forEach" all back
to a single "loop" concept node.

Usage:
    from epichat.seeding.concept_anchors import seed_concept_anchors, CONCEPT_KEYWORD_MAP
    anchor_ids = seed_concept_anchors(kg)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, RelationType, Source
from epichat.core.knowledge_graph import KnowledgeGraph

_SOURCE = Source(name="EPISTEME Universal Concepts", reliability_score=0.99)

# (concept_key, proposition, domain, keywords)
_CONCEPT_DEFINITIONS: List[Tuple[str, str, str, List[str]]] = [
    # ── Control flow ──────────────────────────────────────────────────────────
    ("loop",
     "Loop: a control-flow construct that repeats a block of code while a condition holds "
     "or over a sequence of items. Forms include for, while, do-while, and foreach.",
     "programming_concepts",
     ["loop", "for loop", "while loop", "iteration", "iterate", "foreach", "do while",
      "repeat", "looping", "for each", "range", "enumerate"]),

    ("conditional",
     "Conditional: a control-flow construct that executes different code branches "
     "based on a boolean condition. Forms include if/else, switch/case, and ternary.",
     "programming_concepts",
     ["if", "else", "conditional", "branch", "switch", "case", "ternary", "guard",
      "when", "unless", "boolean condition", "if-else", "if/else"]),

    ("exception_handling",
     "Exception handling: a mechanism to detect and respond to runtime errors by "
     "separating error-handling code from the main logic using try/catch/finally blocks.",
     "programming_concepts",
     ["exception", "try", "catch", "finally", "throw", "raise", "error handling",
      "try-catch", "try/catch", "try/except", "recover", "panic", "rescue"]),

    ("pattern_matching",
     "Pattern matching: a mechanism to check a value against a pattern and "
     "destructure it simultaneously; more powerful than switch/case.",
     "programming_concepts",
     ["pattern matching", "match", "when clause", "destructure", "is expression",
      "instanceof pattern", "sealed class", "discriminated union"]),

    ("recursion",
     "Recursion: a function that calls itself to solve a smaller sub-problem, "
     "with a base case to terminate. Trades stack space for code elegance.",
     "programming_concepts",
     ["recursion", "recursive", "base case", "call itself", "tail recursion",
      "tail call", "memoization", "recursive function", "self-call"]),

    # ── Functions ─────────────────────────────────────────────────────────────
    ("function",
     "Function: a named, reusable block of code that takes inputs (parameters) "
     "and returns an output; the primary unit of abstraction in most languages.",
     "programming_concepts",
     ["function", "method", "procedure", "subroutine", "def ", "fun ", "func ",
      "return value", "parameter", "argument", "call", "invoke"]),

    ("lambda",
     "Lambda / anonymous function: an unnamed function expression, typically used "
     "for short callbacks or functional-style operations.",
     "programming_concepts",
     ["lambda", "anonymous function", "arrow function", "function expression",
      "closure expression", "fun {", "->", "=>", "fn(", "Func<"]),

    ("closure",
     "Closure: a function that captures variables from its enclosing lexical scope, "
     "allowing those variables to outlive the scope in which they were defined.",
     "programming_concepts",
     ["closure", "captures", "lexical scope", "upvalue", "free variable",
      "enclosed variable", "captured variable", "capture list"]),

    ("generator",
     "Generator / coroutine: a function that can pause execution and yield values "
     "one at a time, enabling lazy sequences and cooperative multitasking.",
     "programming_concepts",
     ["generator", "yield", "sequence", "lazy", "suspend", "resume", "coroutine",
      "iterator protocol", "yield return", "function*", "sequence builder"]),

    ("higher_order_function",
     "Higher-order function: a function that takes other functions as arguments "
     "or returns a function, enabling map, filter, reduce, and composition.",
     "programming_concepts",
     ["higher-order", "map", "filter", "reduce", "fold", "flatMap", "compose",
      "function as argument", "callback", "predicate", "higher order"]),

    ("decorator_annotation",
     "Decorator / annotation: metadata attached to a class, method, or field that "
     "modifies behavior at compile-time or runtime without changing the source.",
     "programming_concepts",
     ["decorator", "annotation", "attribute", "@property", "@staticmethod",
      "@Override", "@dataclass", "metadata", "AOP", "aspect"]),

    # ── Data types ────────────────────────────────────────────────────────────
    ("string",
     "String: an immutable sequence of characters used to represent text; "
     "supports operations like concatenation, slicing, formatting, and regex.",
     "programming_concepts",
     ["string", "str", "String", "text", "character", "substring", "concat",
      "interpolation", "format", "template literal", "varchar", "bytes"]),

    ("array_list",
     "Array / List: an ordered, indexed collection of elements; arrays are fixed-size "
     "with O(1) access, dynamic lists grow automatically.",
     "programming_concepts",
     ["array", "list", "ArrayList", "vector", "sequence", "index", "slice",
      "append", "push", "pop", "element", "collection", "MutableList"]),

    ("dictionary_map",
     "Dictionary / Map / HashMap: a key-value store providing O(1) average lookup; "
     "keys must be hashable and unique.",
     "programming_concepts",
     ["dictionary", "dict", "map", "HashMap", "hashtable", "key-value", "lookup",
      "associative", "get(", "put(", "MutableMap", "object literal", "Record"]),

    ("set",
     "Set: an unordered collection of unique elements; provides O(1) membership test "
     "and supports union, intersection, and difference operations.",
     "programming_concepts",
     ["set", "Set", "HashSet", "unique", "union", "intersection", "difference",
      "membership", "contains", "add(", "discard", "frozenset"]),

    ("tuple",
     "Tuple: an immutable, ordered, fixed-length collection of heterogeneous values; "
     "often used to return multiple values from a function.",
     "programming_concepts",
     ["tuple", "Pair", "Triple", "destructure", "unpack", "namedtuple",
      "record type", "value type", "struct", "product type"]),

    # ── OOP ───────────────────────────────────────────────────────────────────
    ("class_object",
     "Class and object: a class is a blueprint defining fields and methods; "
     "an object is an instance of a class holding its own state.",
     "programming_concepts",
     ["class", "object", "instance", "instantiate", "new ", "constructor",
      "self", "this", "field", "member", "property", "attribute"]),

    ("inheritance",
     "Inheritance: a mechanism where a subclass acquires fields and methods from "
     "a superclass, enabling code reuse and is-a relationships.",
     "programming_concepts",
     ["inherit", "extends", "subclass", "superclass", "parent class", "base class",
      "override", "super(", "is-a", "derived", "child class", "open class"]),

    ("interface_protocol",
     "Interface / Protocol / Trait: a contract specifying method signatures without "
     "implementation; classes that implement it must provide the methods.",
     "programming_concepts",
     ["interface", "protocol", "trait", "implements", "abstract", "contract",
      "implement", "conformance", "satisfy", "duck typing", "structural typing"]),

    ("polymorphism",
     "Polymorphism: the ability to treat objects of different types through a common "
     "interface; includes subtype (runtime) and parametric (compile-time) forms.",
     "programming_concepts",
     ["polymorphism", "override", "overload", "virtual", "dispatch", "dynamic",
      "runtime type", "polymorphic", "subtype", "covariance", "contravariance"]),

    ("encapsulation",
     "Encapsulation: bundling data and methods together while hiding internal state "
     "behind a public interface to enforce invariants and reduce coupling.",
     "programming_concepts",
     ["encapsulation", "private", "public", "protected", "access modifier",
      "visibility", "getter", "setter", "accessor", "information hiding"]),

    # ── Async / Concurrency ───────────────────────────────────────────────────
    ("async_await",
     "Async/await: syntax for writing asynchronous code that looks synchronous; "
     "suspends a coroutine until a future completes without blocking the thread.",
     "programming_concepts",
     ["async", "await", "suspend", "coroutine", "non-blocking", "asynchronous",
      "async/await", "async function", "asyncio", "CompletableFuture"]),

    ("promise_future",
     "Promise / Future: an object representing a value that will be available "
     "asynchronously; supports chaining via then/catch or await.",
     "programming_concepts",
     ["promise", "future", "Deferred", "then(", "catch(", "resolve", "reject",
      "CompletableFuture", "Task<", "Mono", "Flux", "RxJava", "Observable"]),

    ("thread",
     "Thread: the smallest unit of CPU execution within a process; multiple threads "
     "share memory and can run concurrently on multi-core processors.",
     "programming_concepts",
     ["thread", "Thread", "pthread", "goroutine", "worker", "executor",
      "ThreadPool", "Runnable", "start()", "join()", "daemon", "fiber"]),

    ("mutex_lock",
     "Mutex / Lock: a synchronization primitive that ensures only one thread accesses "
     "a critical section at a time, preventing data races.",
     "programming_concepts",
     ["mutex", "lock", "synchronized", "monitor", "critical section",
      "acquire", "release", "reentrant", "RWLock", "@Synchronized", "withLock"]),

    ("channel",
     "Channel: a concurrency primitive for safe communication between goroutines or "
     "coroutines by passing values through a typed buffer.",
     "programming_concepts",
     ["channel", "chan ", "send", "receive", "buffer", "select {", "actor",
      "mailbox", "queue", "pipe", "CSP", "message passing"]),

    # ── Memory ────────────────────────────────────────────────────────────────
    ("garbage_collection",
     "Garbage collection: automatic memory management that reclaims memory occupied "
     "by objects no longer reachable by the program.",
     "programming_concepts",
     ["garbage collection", "GC", "reference counting", "mark-and-sweep",
      "generational", "heap", "finalize", "destructor", "dealloc", "ARC",
      "retain cycle", "weak reference", "soft reference"]),

    ("memory_management",
     "Memory management: controlling allocation and deallocation of memory; "
     "includes stack vs heap allocation, RAII, and ownership models.",
     "programming_concepts",
     ["memory", "heap", "stack", "allocation", "deallocation", "RAII", "ownership",
      "borrow", "lifetime", "smart pointer", "unique_ptr", "shared_ptr",
      "malloc", "free", "new", "delete", "unsafe"]),

    # ── Type system ───────────────────────────────────────────────────────────
    ("type_system",
     "Type system: a set of rules that assigns types to expressions; "
     "static typing catches errors at compile time, dynamic typing at runtime.",
     "programming_concepts",
     ["type", "static typing", "dynamic typing", "type inference", "typed",
      "type annotation", "type hint", "generic", "parameterized", "coerce",
      "cast", "type check", "mypy", "TypeScript", "type safety"]),

    ("generics_templates",
     "Generics / Templates: parameterized types that let functions and data structures "
     "operate on any type while maintaining type safety.",
     "programming_concepts",
     ["generic", "template", "type parameter", "<T>", "parameterized", "reify",
      "bounded type", "variance", "wildcard", "phantom type", "type constraint"]),

    ("null_handling",
     "Null / None handling: strategies to represent missing values safely, including "
     "null checks, Optional types, nullable types, and the null object pattern.",
     "programming_concepts",
     ["null", "None", "nil", "nullable", "Optional", "Maybe", "null check",
      "?.","?:", "NullPointerException", "null safety", "unwrap", "ifPresent",
      "null coalescing", "non-null"]),

    # ── Modules / Packages ────────────────────────────────────────────────────
    ("module_system",
     "Module / Package system: a mechanism to organize code into separate namespaces "
     "and control what is exported and imported between files.",
     "programming_concepts",
     ["module", "package", "namespace", "import", "export", "require(", "from ",
      "use ", "include", "dependency", "visibility", "public API", "internal"]),

    # ── Functional programming ────────────────────────────────────────────────
    ("immutability",
     "Immutability: a property where a value cannot be modified after creation; "
     "enables safer concurrency, easier reasoning, and predictable behavior.",
     "programming_concepts",
     ["immutable", "immutability", "const", "val ", "final", "frozen", "readonly",
      "pure", "persistent data structure", "copy-on-write", "freeze"]),

    ("functional_composition",
     "Functional composition: combining simple functions to build complex behavior; "
     "includes map, filter, reduce, and function pipelines.",
     "programming_concepts",
     ["compose", "pipeline", "pipe", "chain", "flatMap", "map(", "filter(",
      "reduce(", "fold", "functional", "pure function", "point-free"]),

    # ── Testing ───────────────────────────────────────────────────────────────
    ("unit_testing",
     "Unit testing: testing individual functions or methods in isolation to verify "
     "they behave correctly for all expected inputs and edge cases.",
     "programming_concepts",
     ["unit test", "test", "assert", "assertEquals", "pytest", "JUnit", "Jest",
      "spec", "mock", "stub", "fixture", "test suite", "TDD", "test case"]),

    # ── Serialization ─────────────────────────────────────────────────────────
    ("serialization",
     "Serialization: converting an object's state to a storable or transmittable "
     "format (JSON, XML, binary); deserialization reverses the process.",
     "programming_concepts",
     ["serialize", "deserialize", "JSON", "XML", "marshal", "unmarshal", "encode",
      "decode", "pickle", "protobuf", "Parcelable", "Codable", "parse"]),

    # ── I/O ───────────────────────────────────────────────────────────────────
    ("io_streams",
     "I/O streams: abstractions for reading from and writing to files, network "
     "sockets, and other data sources in a uniform, buffered way.",
     "programming_concepts",
     ["stream", "read", "write", "file", "IO", "InputStream", "OutputStream",
      "buffer", "flush", "close", "open(", "with open", "FileReader",
      "BufferedReader", "readline"]),

    # ── Error types ───────────────────────────────────────────────────────────
    ("result_type",
     "Result / Either type: a functional alternative to exceptions that encodes "
     "success or failure as a value, forcing callers to handle both cases.",
     "programming_concepts",
     ["Result", "Either", "Ok(", "Err(", "Success", "Failure", "Try ", "Left",
      "Right", "fold(", "getOrElse", "onSuccess", "onFailure", "runCatching"]),
]

# Maps concept_key -> list of keywords (for fast matching)
CONCEPT_KEYWORD_MAP: Dict[str, List[str]] = {
    key: kws for key, _, _, kws in _CONCEPT_DEFINITIONS
}


def seed_concept_anchors(kg: KnowledgeGraph) -> Dict[str, str]:
    """
    Add concept anchor EUs to the knowledge graph.
    Returns {concept_key: eu_id} for all anchors (new or already existing).
    """
    anchor_ids: Dict[str, str] = {}

    for key, proposition, domain, keywords in _CONCEPT_DEFINITIONS:
        # Check if anchor already exists via semantic search
        existing = kg.find_similar(proposition, top_k=1, threshold=0.92)
        if existing and existing[0].domain == "programming_concepts":
            anchor_ids[key] = existing[0].id
            continue

        eu = EpistemicUnit(
            proposition=proposition,
            knowledge_type=KnowledgeType.AXIOM,
            confidence=0.98,
            domain=domain,
            sources=[_SOURCE],
            keywords=keywords,
        )
        kg.add(eu)
        anchor_ids[key] = eu.id

    added = sum(1 for k in anchor_ids if k not in {
        e.domain for e in kg.units.values() if e.domain == "programming_concepts"
    })
    print(f"[ConceptAnchors] {len(anchor_ids)} concept anchors ready", flush=True)
    return anchor_ids
