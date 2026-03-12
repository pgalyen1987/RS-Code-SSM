#!/usr/bin/env python3
"""
EPISTEME — Epistemology-Based Knowledge System
----------------------------------------------
Usage:
  python main.py               # Launch interactive shell (load existing graph)
  python main.py --seed        # Seed knowledge from Wikipedia + web sources
  python main.py --seed-quick  # Seed a small subset (fast, for testing)
  python main.py --query "..."  # One-shot query, print result and exit
  python main.py --stats       # Print knowledge graph stats and exit
"""

import argparse
import sys


def build_graph(load_existing: bool = True):
    from core.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    if load_existing:
        kg.load()
    return kg


def cmd_seed(quick: bool = False):
    from core.knowledge_graph import KnowledgeGraph
    from seeding.seeder_pipeline import SeedPipeline, DEFAULT_TOPICS

    kg = build_graph(load_existing=True)
    pipeline = SeedPipeline(kg)

    if quick:
        quick_topics = [
            "Algorithm", "Data structure", "Big O notation",
            "Object-oriented programming", "Design Patterns (book)",
            "Kotlin (programming language)", "Python (programming language)",
            "Transmission Control Protocol", "Computer security", "OWASP",
        ]
        pipeline.run(topics=quick_topics, max_sentences=20,
                     include_web=False, save_after=True)
    else:
        pipeline.run(save_after=True)


def cmd_query(question: str):
    from language.query_engine import QueryEngine
    from rich.console import Console

    console = Console()
    kg = build_graph()
    engine = QueryEngine(kg)

    with console.status("Querying…"):
        result = engine.query(question)

    console.print(result.summary())
    if result.gaps:
        console.print(f"\nGaps: {result.gaps}")


def cmd_stats():
    from rich import print as rprint
    kg = build_graph()
    rprint(kg.stats())


def cmd_shell():
    from interface.cli import EpistemeShell
    kg = build_graph()
    shell = EpistemeShell(kg)
    shell.run()


def main():
    parser = argparse.ArgumentParser(
        description="EPISTEME — Justified Belief Knowledge System"
    )
    parser.add_argument("--seed",       action="store_true", help="Seed full knowledge base")
    parser.add_argument("--seed-quick", action="store_true", help="Seed small subset (fast)")
    parser.add_argument("--query",      type=str,            help="One-shot query")
    parser.add_argument("--stats",      action="store_true", help="Print graph stats")

    args = parser.parse_args()

    if args.seed:
        cmd_seed(quick=False)
    elif args.seed_quick:
        cmd_seed(quick=True)
    elif args.query:
        cmd_query(args.query)
    elif args.stats:
        cmd_stats()
    else:
        cmd_shell()


if __name__ == "__main__":
    main()
