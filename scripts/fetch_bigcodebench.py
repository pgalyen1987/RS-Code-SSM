"""Fetch BigCodeBench from HF and filter for problems usable with installed libs."""
import ast
import json
import sys

SAFE_LIBS = {
    # stdlib
    'os', 'sys', 'math', 're', 'json', 'random', 'string', 'time', 'datetime',
    'collections', 'itertools', 'functools', 'operator', 'typing', 'pathlib',
    'io', 'abc', 'copy', 'hashlib', 'struct', 'shutil', 'glob', 'heapq',
    'bisect', 'queue', 'threading', 'multiprocessing', 'subprocess', 'tempfile',
    'contextlib', 'dataclasses', 'enum', 'statistics', 'decimal', 'fractions',
    'unittest', 'textwrap', 'inspect', 'ast', 'base64', 'binascii', 'csv',
    'sqlite3', 'uuid', 'hmac', 'zlib', 'gzip', 'tarfile', 'zipfile',
    'urllib', 'http', 'socket', 'argparse', 'logging', 'warnings', 'pprint',
    # installed scientific libs
    'numpy', 'np', 'pandas', 'pd', 'scipy', 'sklearn', 'requests',
}

def parse_libs(raw):
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []

def is_safe(libs_raw):
    libs = parse_libs(libs_raw)
    return all(lib.split('.')[0] in SAFE_LIBS for lib in libs)

def main():
    from datasets import load_dataset
    print("Loading BigCodeBench from HuggingFace...", flush=True)
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", trust_remote_code=True)
    print(f"Total problems: {len(ds)}", flush=True)

    kept = []
    lib_counter = {}
    for row in ds:
        libs = parse_libs(row.get('libs', '[]'))
        for lib in libs:
            lib_counter[lib] = lib_counter.get(lib, 0) + 1
        if is_safe(libs):
            # build test code — BigCodeBench uses 'test' field with unittest
            test_code = row.get('test', '')
            code_prompt = row.get('complete_prompt', row.get('instruct_prompt', ''))
            canonical = row.get('canonical_solution', '')
            if not test_code or not code_prompt:
                continue
            # wrap test so it runs standalone
            full_test = f"{canonical}\n\n{test_code}\n\nimport unittest\nunittest.main(exit=False, verbosity=0)"
            kept.append({
                'prompt': code_prompt.strip(),
                'test_code': full_test,
                'language': 'python',
                'source': 'bigcodebench',
                'entry_point': row.get('entry_point', ''),
            })

    # top filtered libs
    top_filtered = sorted(
        [(k, v) for k, v in lib_counter.items() if k.split('.')[0] not in SAFE_LIBS],
        key=lambda x: -x[1]
    )[:10]
    print(f"Kept: {len(kept)} problems (filtered {len(ds)-len(kept)})", flush=True)
    print(f"Top filtered libs: {top_filtered}", flush=True)

    out = '/workspace/RS-Code-SSM/data/grpo_bigcodebench.jsonl'
    with open(out, 'w') as f:
        for rec in kept:
            f.write(json.dumps(rec) + '\n')
    print(f"Written to {out}", flush=True)

if __name__ == '__main__':
    main()
