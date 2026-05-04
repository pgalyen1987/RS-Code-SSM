"""
Curated entry points to vendor / standards documentation used to seed EpistemicUnits.

Pages are chosen where HTML is typically server-rendered (MDN, docs.python.org,
nodejs.org/dist/.../docs/api, cppreference, Oracle Java tutorials, kotlinlang.org,
Microsoft Learn). JS-heavy SPAs may yield fewer paragraphs; failures are skipped.

License / attribution: content remains © respective vendors; we store short excerpts
with Source.url pointing to the canonical page (fair use for RAG indexing).
"""
from __future__ import annotations

from typing import Any, Dict, List

# reliability: authority of the publisher for epistemic scoring
OFFICIAL_DOCUMENTATION_SOURCES: List[Dict[str, Any]] = [
    # --- JavaScript (ECMAScript + Web APIs via MDN) ---
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
        "name": "MDN — JavaScript Guide",
        "domain": "javascript",
        "language": "JavaScript",
        "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise",
        "name": "MDN — Promise",
        "domain": "javascript",
        "language": "JavaScript",
        "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions",
        "name": "MDN — Functions",
        "domain": "javascript",
        "language": "JavaScript",
        "reliability": 0.93,
    },
    # --- Node.js (built-in HTML API docs under dist/) ---
    {
        "url": "https://nodejs.org/dist/latest/docs/api/synopsis.html",
        "name": "Node.js — Synopsis",
        "domain": "nodejs",
        "language": "JavaScript",
        "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/modules.html",
        "name": "Node.js — Modules (CommonJS)",
        "domain": "nodejs",
        "language": "JavaScript",
        "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/esm.html",
        "name": "Node.js — ECMAScript modules",
        "domain": "nodejs",
        "language": "JavaScript",
        "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/fs.html",
        "name": "Node.js — fs",
        "domain": "nodejs",
        "language": "JavaScript",
        "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/http.html",
        "name": "Node.js — HTTP",
        "domain": "nodejs",
        "language": "JavaScript",
        "reliability": 0.94,
    },
    # --- React (may be partially hydrated; still worth trying) ---
    {
        "url": "https://react.dev/learn/thinking-in-react",
        "name": "React — Thinking in React",
        "domain": "react",
        "language": "JavaScript",
        "reliability": 0.92,
    },
    {
        "url": "https://react.dev/learn/state-a-components-memory",
        "name": "React — State",
        "domain": "react",
        "language": "JavaScript",
        "reliability": 0.92,
    },
    {
        "url": "https://react.dev/reference/react/hooks",
        "name": "React — Hooks reference overview",
        "domain": "react",
        "language": "JavaScript",
        "reliability": 0.92,
    },
    # --- Python (docs.python.org) ---
    {
        "url": "https://docs.python.org/3/tutorial/index.html",
        "name": "Python Tutorial — overview",
        "domain": "python",
        "language": "Python",
        "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/tutorial/introduction.html",
        "name": "Python Tutorial — Introduction",
        "domain": "python",
        "language": "Python",
        "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/functions.html",
        "name": "Python — Built-in functions",
        "domain": "python",
        "language": "Python",
        "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/stdtypes.html",
        "name": "Python — Built-in types",
        "domain": "python",
        "language": "Python",
        "reliability": 0.94,
    },
    # --- C++ (cppreference) ---
    {
        "url": "https://en.cppreference.com/w/cpp/language/function",
        "name": "cppreference — Functions",
        "domain": "cpp",
        "language": "C++",
        "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/language/classes",
        "name": "cppreference — Classes",
        "domain": "cpp",
        "language": "C++",
        "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/language/raii",
        "name": "cppreference — RAII",
        "domain": "cpp",
        "language": "C++",
        "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/language/templates",
        "name": "cppreference — Templates",
        "domain": "cpp",
        "language": "C++",
        "reliability": 0.91,
    },
    # --- Java (Oracle tutorials) ---
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/concepts/index.html",
        "name": "Oracle — Java concepts",
        "domain": "java",
        "language": "Java",
        "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/javaOO/index.html",
        "name": "Oracle — Classes and objects",
        "domain": "java",
        "language": "Java",
        "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/data/index.html",
        "name": "Oracle — Numbers and strings",
        "domain": "java",
        "language": "Java",
        "reliability": 0.93,
    },
    # --- Kotlin ---
    {
        "url": "https://kotlinlang.org/docs/basic-syntax.html",
        "name": "Kotlin — Basic syntax",
        "domain": "kotlin",
        "language": "Kotlin",
        "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/classes.html",
        "name": "Kotlin — Classes",
        "domain": "kotlin",
        "language": "Kotlin",
        "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/functions.html",
        "name": "Kotlin — Functions",
        "domain": "kotlin",
        "language": "Kotlin",
        "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/coroutines-overview.html",
        "name": "Kotlin — Coroutines overview",
        "domain": "kotlin",
        "language": "Kotlin",
        "reliability": 0.93,
    },
    # --- C# (Microsoft Learn) ---
    {
        "url": "https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/",
        "name": "Microsoft Learn — Tour of C#",
        "domain": "csharp",
        "language": "C#",
        "reliability": 0.93,
    },
    {
        "url": "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/",
        "name": "Microsoft Learn — Types",
        "domain": "csharp",
        "language": "C#",
        "reliability": 0.93,
    },
    {
        "url": "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/object-oriented/",
        "name": "Microsoft Learn — OOP in C#",
        "domain": "csharp",
        "language": "C#",
        "reliability": 0.93,
    },
    {
        "url": "https://learn.microsoft.com/en-us/dotnet/csharp/asynchronous-programming/",
        "name": "Microsoft Learn — Async programming",
        "domain": "csharp",
        "language": "C#",
        "reliability": 0.93,
    },

    # --- Python (expanded) ---
    {
        "url": "https://docs.python.org/3/tutorial/controlflow.html",
        "name": "Python Tutorial — Control flow",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/tutorial/classes.html",
        "name": "Python Tutorial — Classes",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/tutorial/errors.html",
        "name": "Python Tutorial — Errors and exceptions",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/tutorial/datastructures.html",
        "name": "Python Tutorial — Data structures",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/asyncio.html",
        "name": "Python — asyncio",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/concurrent.futures.html",
        "name": "Python — concurrent.futures",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/collections.html",
        "name": "Python — collections",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/itertools.html",
        "name": "Python — itertools",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/functools.html",
        "name": "Python — functools",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/reference/expressions.html",
        "name": "Python Reference — Expressions",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/reference/compound_stmts.html",
        "name": "Python Reference — Compound statements",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/typing.html",
        "name": "Python — typing module",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },
    {
        "url": "https://docs.python.org/3/library/unittest.html",
        "name": "Python — unittest",
        "domain": "python", "language": "Python", "reliability": 0.94,
    },

    # --- C++ (cppreference expanded) ---
    {
        "url": "https://en.cppreference.com/w/cpp/language/range-for",
        "name": "cppreference — Range-for loop",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/language/lambda",
        "name": "cppreference — Lambda expressions",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/language/exceptions",
        "name": "cppreference — Exceptions",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/language/move_semantics",
        "name": "cppreference — Move semantics",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/memory",
        "name": "cppreference — Memory management",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/container",
        "name": "cppreference — STL containers",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/algorithm",
        "name": "cppreference — STL algorithms",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },
    {
        "url": "https://en.cppreference.com/w/cpp/thread",
        "name": "cppreference — Thread support",
        "domain": "cpp", "language": "C++", "reliability": 0.91,
    },

    # --- Java (expanded Oracle tutorials) ---
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/nutsandbolts/flow.html",
        "name": "Oracle — Control flow",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/generics/index.html",
        "name": "Oracle — Generics",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/javaOO/lambdaexpressions.html",
        "name": "Oracle — Lambda expressions",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/essential/exceptions/index.html",
        "name": "Oracle — Exceptions",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/collections/index.html",
        "name": "Oracle — Collections",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/essential/concurrency/index.html",
        "name": "Oracle — Concurrency",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/annotations/index.html",
        "name": "Oracle — Annotations",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },
    {
        "url": "https://docs.oracle.com/javase/tutorial/java/IandI/index.html",
        "name": "Oracle — Interfaces and inheritance",
        "domain": "java", "language": "Java", "reliability": 0.93,
    },

    # --- Kotlin (expanded) ---
    {
        "url": "https://kotlinlang.org/docs/control-flow.html",
        "name": "Kotlin — Control flow",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/exceptions.html",
        "name": "Kotlin — Exceptions",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/collections-overview.html",
        "name": "Kotlin — Collections",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/lambdas.html",
        "name": "Kotlin — Lambdas",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/sealed-classes.html",
        "name": "Kotlin — Sealed classes",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/generics.html",
        "name": "Kotlin — Generics",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/extensions.html",
        "name": "Kotlin — Extension functions",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/null-safety.html",
        "name": "Kotlin — Null safety",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/data-classes.html",
        "name": "Kotlin — Data classes",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/delegation.html",
        "name": "Kotlin — Delegation",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },
    {
        "url": "https://kotlinlang.org/docs/flow.html",
        "name": "Kotlin — Flow",
        "domain": "kotlin", "language": "Kotlin", "reliability": 0.93,
    },

    # --- JavaScript / MDN expanded ---
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_generators",
        "name": "MDN — Iterators and generators",
        "domain": "javascript", "language": "JavaScript", "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Classes",
        "name": "MDN — Classes",
        "domain": "javascript", "language": "JavaScript", "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Inheritance_and_the_prototype_chain",
        "name": "MDN — Prototype chain",
        "domain": "javascript", "language": "JavaScript", "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Control_flow_and_error_handling",
        "name": "MDN — Control flow and error handling",
        "domain": "javascript", "language": "JavaScript", "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Closures",
        "name": "MDN — Closures",
        "domain": "javascript", "language": "JavaScript", "reliability": 0.93,
    },
    {
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array",
        "name": "MDN — Array",
        "domain": "javascript", "language": "JavaScript", "reliability": 0.93,
    },

    # --- Node.js (expanded) ---
    {
        "url": "https://nodejs.org/dist/latest/docs/api/events.html",
        "name": "Node.js — Events",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/stream.html",
        "name": "Node.js — Streams",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/buffer.html",
        "name": "Node.js — Buffer",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/path.html",
        "name": "Node.js — Path",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/async_hooks.html",
        "name": "Node.js — Async hooks",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/worker_threads.html",
        "name": "Node.js — Worker threads",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },
    {
        "url": "https://nodejs.org/dist/latest/docs/api/crypto.html",
        "name": "Node.js — Crypto",
        "domain": "nodejs", "language": "JavaScript", "reliability": 0.94,
    },

    # --- React (expanded) ---
    {
        "url": "https://react.dev/learn/managing-state",
        "name": "React — Managing state",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },
    {
        "url": "https://react.dev/learn/escape-hatches",
        "name": "React — Escape hatches (refs, effects)",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },
    {
        "url": "https://react.dev/reference/react/Component",
        "name": "React — Component",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },
    {
        "url": "https://react.dev/reference/react/useEffect",
        "name": "React — useEffect",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },
    {
        "url": "https://react.dev/reference/react/useContext",
        "name": "React — useContext",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },
    {
        "url": "https://react.dev/reference/react/useReducer",
        "name": "React — useReducer",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },
    {
        "url": "https://react.dev/learn/rendering-lists",
        "name": "React — Rendering lists",
        "domain": "react", "language": "JavaScript", "reliability": 0.92,
    },

    # --- Rust (doc.rust-lang.org) ---
    {
        "url": "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html",
        "name": "Rust Book — Ownership",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },
    {
        "url": "https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html",
        "name": "Rust Book — Borrowing",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },
    {
        "url": "https://doc.rust-lang.org/book/ch06-00-enums.html",
        "name": "Rust Book — Enums and pattern matching",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },
    {
        "url": "https://doc.rust-lang.org/book/ch10-00-generics.html",
        "name": "Rust Book — Generics and traits",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },
    {
        "url": "https://doc.rust-lang.org/book/ch16-00-concurrency.html",
        "name": "Rust Book — Concurrency",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },
    {
        "url": "https://doc.rust-lang.org/book/ch09-00-error-handling.html",
        "name": "Rust Book — Error handling",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },
    {
        "url": "https://doc.rust-lang.org/book/ch13-00-functional-features.html",
        "name": "Rust Book — Functional features",
        "domain": "rust", "language": "Rust", "reliability": 0.94,
    },

    # --- Go (go.dev/doc) ---
    {
        "url": "https://go.dev/doc/effective_go",
        "name": "Go — Effective Go",
        "domain": "go", "language": "Go", "reliability": 0.94,
    },
    {
        "url": "https://go.dev/tour/concurrency/1",
        "name": "Go Tour — Goroutines",
        "domain": "go", "language": "Go", "reliability": 0.93,
    },
    {
        "url": "https://pkg.go.dev/sync",
        "name": "Go — sync package",
        "domain": "go", "language": "Go", "reliability": 0.93,
    },
    {
        "url": "https://pkg.go.dev/context",
        "name": "Go — context package",
        "domain": "go", "language": "Go", "reliability": 0.93,
    },
]
