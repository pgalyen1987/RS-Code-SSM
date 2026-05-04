#!/usr/bin/env python3
"""
Scale EpiChat toward ~100k EpistemicUnits.

Approach:
  1) Wikipedia: seed titles from epichat/seeding/wiki_mass_topics.py (deduped vs scripts/generate_eus.py).
  2) MDN bulk: Global_Objects + extra guide URLs (epichat/seeding/mdn_js_bulk.py).

Expect many merges via semantic dedup; reaching exactly 100k may take multiple runs or higher
--max-sentences. Also run scripts/generate_eus.py Phase 2 (LLM) if Ollama is available.

Example:
  cd /home/me/SSM && source scripts/env.sh
  python scripts/scale_epichat_graph.py --target 100000 --max-sentences 65 2>&1 | tee logs/kg_scale.log
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
REPO_ROOT = os.environ.get("REPO_ROOT") or str(_repo)
EPICHAT_DIR = os.environ.get("EPICHAT_DIR") or str(_repo / "epichat")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from epichat.core.knowledge_graph import KnowledgeGraph
from epichat.seeding.code_seeder import CodeSeeder
from epichat.seeding.concept_anchors import seed_concept_anchors
from epichat.seeding.concept_linker import link_concepts
from epichat.seeding.mdn_js_bulk import combined_mdn_extra_sources
from epichat.seeding.official_docs_registry import OFFICIAL_DOCUMENTATION_SOURCES
from epichat.seeding.official_docs_seeder import OfficialDocsSeeder
from epichat.seeding.wiki_mass_topics import MASS_ADDITIONAL_TOPICS


def load_base_wiki_topics() -> set[str]:
    p = Path(REPO_ROOT) / "scripts" / "generate_eus.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "WIKI_TOPICS":
                    return {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale EpiChat knowledge graph.")
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--max-sentences", type=int, default=65)
    parser.add_argument("--wiki-chunk-size", type=int, default=200)
    parser.add_argument("--skip-wikipedia-mass", action="store_true")
    parser.add_argument("--skip-mdn-bulk", action="store_true")
    parser.add_argument("--docs-max-paragraphs", type=int, default=28)
    parser.add_argument("--docs-delay", type=float, default=0.28)
    parser.add_argument(
        "--include-base-wikipedia",
        action="store_true",
        help="Re-seed scripts/generate_eus.py WIKI_TOPICS with higher sentence cap.",
    )
    args = parser.parse_args()

    epichat_root = Path(EPICHAT_DIR).resolve()
    data = epichat_root / "episteme_data"
    data.mkdir(parents=True, exist_ok=True)

    print(f"[scale] EPICHAT_DIR={epichat_root}", flush=True)
    print(f"[scale] episteme_data={data}", flush=True)

    kg = KnowledgeGraph()
    if (data / "units.json").exists():
        kg.load(str(data))
    start = len(kg.units)
    print(f"[scale] Starting units: {start}  target: {args.target}", flush=True)

    base = load_base_wiki_topics()
    new_topics = [t for t in MASS_ADDITIONAL_TOPICS if t not in base]
    print(
        f"[scale] Mass Wikipedia titles (excluding generate_eus base): {len(new_topics)}",
        flush=True,
    )

    seeder = CodeSeeder(kg)

    if args.include_base_wikipedia and len(kg.units) < args.target:
        base_list = sorted(load_base_wiki_topics())
        print(f"[scale] Base Wikipedia topics (generate_eus): {len(base_list)}", flush=True)
        for i in range(0, len(base_list), args.wiki_chunk_size):
            if len(kg.units) >= args.target:
                break
            chunk = base_list[i : i + args.wiki_chunk_size]
            before = len(kg.units)
            seeder.seed_wikipedia(topics=chunk, max_sentences=args.max_sentences)
            after = len(kg.units)
            print(
                f"[scale] Base wiki chunk {i // args.wiki_chunk_size + 1}: "
                f"+{after - before} → total {after}",
                flush=True,
            )
            kg.save(str(data))

    if not args.skip_wikipedia_mass:
        for i in range(0, len(new_topics), args.wiki_chunk_size):
            if len(kg.units) >= args.target:
                print("[scale] Target reached — stopping Wikipedia chunks.", flush=True)
                break
            chunk = new_topics[i : i + args.wiki_chunk_size]
            before = len(kg.units)
            seeder.seed_wikipedia(topics=chunk, max_sentences=args.max_sentences)
            after = len(kg.units)
            print(
                f"[scale] Wiki chunk {i // args.wiki_chunk_size + 1}: "
                f"+{after - before} → total {after}",
                flush=True,
            )
            kg.save(str(data))

    # ── Official docs (Python, C++, Java, Kotlin, JS, Node, React, Rust, Go) ──
    if len(kg.units) < args.target:
        before = len(kg.units)
        OfficialDocsSeeder(kg).seed(
            sources=OFFICIAL_DOCUMENTATION_SOURCES,
            max_paragraphs_per_url=args.docs_max_paragraphs,
            delay_sec=args.docs_delay,
        )
        after = len(kg.units)
        print(f"[scale] Official docs: +{after - before} → total {after}", flush=True)
        kg.save(str(data))

    if (
        not args.skip_mdn_bulk
        and len(kg.units) < args.target
    ):
        before = len(kg.units)
        OfficialDocsSeeder(kg).seed(
            sources=combined_mdn_extra_sources(),
            max_paragraphs_per_url=args.docs_max_paragraphs,
            delay_sec=args.docs_delay,
        )
        after = len(kg.units)
        print(f"[scale] MDN bulk: +{after - before} → total {after}", flush=True)
        kg.save(str(data))

    # ── Concept anchors + linking pass ────────────────────────────────────────
    print("[scale] Seeding concept anchors...", flush=True)
    seed_concept_anchors(kg)
    kg.save(str(data))

    print("[scale] Linking language-specific EUs to concept anchors...", flush=True)
    edges = link_concepts(kg, min_score=2)
    print(f"[scale] Concept edges: {edges}", flush=True)
    kg.save(str(data))

    # ── Upload to HuggingFace ─────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_DATASET_REPO", "pgalyen1987/rs-code-ssm-traces")
    if hf_token:
        print(f"[scale] Uploading to HF dataset repo {hf_repo}...", flush=True)
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            for fname in ["units.json", "faiss.index", "faiss_map.json"]:
                fpath = data / fname
                if fpath.exists():
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=f"epichat/{fname}",
                        repo_id=hf_repo,
                        repo_type="dataset",
                    )
                    print(f"[scale] Uploaded epichat/{fname}", flush=True)
        except Exception as e:
            print(f"[scale] HF upload failed: {e}", flush=True)
    else:
        print("[scale] HF_TOKEN not set — skipping upload", flush=True)

    final = len(kg.units)
    print(
        f"[scale] Done. Units: {final} (+{final - start}). "
        f"Remaining to target: {max(0, args.target - final)}",
        flush=True,
    )
    print(
        "[scale] Export traces: python -u -m train.epichat_export "
        f"--epichat-dir {epichat_root} --output data/epichat_traces.jsonl",
        flush=True,
    )


if __name__ == "__main__":
    main()
