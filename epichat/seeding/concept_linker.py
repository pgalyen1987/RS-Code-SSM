"""
ConceptLinker: post-processing pass that connects language-specific EUs
to universal concept anchors via RelationType.SPECIALIZES.

Example graph edges created:
  "Python for loop iterates over a range" --SPECIALIZES--> "Loop: a control-flow construct..."
  "Java enhanced for loop over Iterable"  --SPECIALIZES--> "Loop: a control-flow construct..."
  "Kotlin coroutine suspend function"     --SPECIALIZES--> "Async/await: syntax for writing..."

Run after all seeding is complete. Safe to run multiple times (won't re-add edges).
"""
from __future__ import annotations

from typing import Dict, List
from epichat.core.epistemic_unit import RelationType
from epichat.core.knowledge_graph import KnowledgeGraph
from .concept_anchors import CONCEPT_KEYWORD_MAP, seed_concept_anchors


def _score(proposition: str, keywords: List[str]) -> int:
    """Count how many concept keywords appear in the proposition (lowercased)."""
    prop_lower = proposition.lower()
    return sum(1 for kw in keywords if kw.lower() in prop_lower)


def link_concepts(kg: KnowledgeGraph, min_score: int = 2) -> int:
    """
    For every non-anchor EU, find matching concept anchors and add SPECIALIZES edges.

    min_score: minimum keyword matches required to create a link (default 2).
    Returns number of new edges added.
    """
    # Ensure anchors exist and get their ids
    anchor_ids = seed_concept_anchors(kg)

    # Build reverse map: concept_key -> (anchor_eu_id, keywords)
    concept_map: Dict[str, tuple] = {
        key: (anchor_ids[key], CONCEPT_KEYWORD_MAP[key])
        for key in anchor_ids
    }

    edges_added = 0
    anchor_id_set = set(anchor_ids.values())

    for eu_id, eu in kg.units.items():
        # Skip concept anchors themselves
        if eu_id in anchor_id_set:
            continue
        # Skip EUs with very short propositions (unlikely to match meaningfully)
        if len(eu.proposition) < 40:
            continue

        for key, (anchor_id, keywords) in concept_map.items():
            score = _score(eu.proposition, keywords)
            if score < min_score:
                continue
            # Check not already linked
            existing = eu.relations.get(RelationType.SPECIALIZES.value, [])
            if anchor_id in existing:
                continue
            # Add SPECIALIZES edge EU -> anchor
            eu.add_relation(RelationType.SPECIALIZES, anchor_id)
            # Add GENERALIZES edge anchor -> EU (bidirectional for graph traversal)
            anchor_eu = kg.units.get(anchor_id)
            if anchor_eu:
                anchor_eu.add_relation(RelationType.GENERALIZES, eu_id)
            # Register edge in NetworkX graph
            if anchor_id in kg.graph:
                kg.graph.add_edge(eu_id, anchor_id, relation=RelationType.SPECIALIZES.value)
            edges_added += 1

    print(f"[ConceptLinker] {edges_added} new concept edges added", flush=True)
    return edges_added
