"""
Additional MDN URLs under JavaScript/Reference/Global_Objects/* for bulk EU seeding.
404s are skipped gracefully by OfficialDocsSeeder.
"""
from __future__ import annotations

from typing import Any, Dict, List

# EcmaScript built-ins / intrinsics commonly documented at Global_Objects/<Name>
_MDN_GLOBAL_OBJECTS: List[str] = [
    "AggregateError",
    "Array",
    "ArrayBuffer",
    "Atomics",
    "BigInt",
    "BigInt64Array",
    "BigUint64Array",
    "Boolean",
    "DataView",
    "Date",
    "decodeURI",
    "decodeURIComponent",
    "encodeURI",
    "encodeURIComponent",
    "Error",
    "escape",
    "eval",
    "EvalError",
    "FinalizationRegistry",
    "Float32Array",
    "Float64Array",
    "Function",
    "globalThis",
    "Infinity",
    "Int8Array",
    "Int16Array",
    "Int32Array",
    "Intl",
    "isFinite",
    "isNaN",
    "JSON",
    "Map",
    "Math",
    "NaN",
    "Number",
    "Object",
    "parseFloat",
    "parseInt",
    "Promise",
    "Proxy",
    "RangeError",
    "ReferenceError",
    "Reflect",
    "RegExp",
    "Set",
    "SharedArrayBuffer",
    "String",
    "Symbol",
    "SyntaxError",
    "TypeError",
    "Uint8Array",
    "Uint8ClampedArray",
    "Uint16Array",
    "Uint32Array",
    "undefined",
    "unescape",
    "URIError",
    "WeakMap",
    "WeakRef",
    "WeakSet",
]


def mdn_global_object_sources(reliability: float = 0.93) -> List[Dict[str, Any]]:
    base = "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects"
    out: List[Dict[str, Any]] = []
    for name in _MDN_GLOBAL_OBJECTS:
        slug = name.replace(" ", "_")
        out.append(
            {
                "url": f"{base}/{slug}",
                "name": f"MDN — JS GlobalObjects.{name}",
                "domain": "javascript",
                "language": "JavaScript",
                "reliability": reliability,
            }
        )
    return out


def mdn_js_guide_sources(reliability: float = 0.93) -> List[Dict[str, Any]]:
    """Extra guide / syntax pages (beyond main registry)."""
    paths = [
        "/en-US/docs/Web/JavaScript/Guide/Loops_and_iteration",
        "/en-US/docs/Web/JavaScript/Guide/Functions",
        "/en-US/docs/Web/JavaScript/Guide/Using_classes",
        "/en-US/docs/Web/JavaScript/Guide/Modules",
        "/en-US/docs/Web/JavaScript/Guide/Regular_expressions",
        "/en-US/docs/Web/JavaScript/Guide/Working_with_objects",
        "/en-US/docs/Web/JavaScript/Reference/Statements/async_function",
        "/en-US/docs/Web/JavaScript/Reference/Statements/function*",
        "/en-US/docs/Web/JavaScript/Reference/Operators/Destructuring_assignment",
        "/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax",
        "/en-US/docs/Web/JavaScript/Reference/Operators/Optional_chaining",
        "/en-US/docs/Web/JavaScript/Reference/Operators/Nullish_coalescing_operator",
    ]
    root = "https://developer.mozilla.org"
    return [
        {
            "url": f"{root}{p}",
            "name": f"MDN — JS {p.split('/')[-1]}",
            "domain": "javascript",
            "language": "JavaScript",
            "reliability": reliability,
        }
        for p in paths
    ]


def combined_mdn_extra_sources() -> List[Dict[str, Any]]:
    return mdn_global_object_sources() + mdn_js_guide_sources()
