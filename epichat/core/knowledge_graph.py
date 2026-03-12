from __future__ import annotations
import json
import os
import pickle
import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Tuple

from .epistemic_unit import EpistemicUnit, KnowledgeType, RelationType


class KnowledgeGraph:
    """
    Sparse graph-based knowledge store.
    - NetworkX DiGraph for structure/traversal
    - FAISS for semantic similarity search
    - Confidence-gated admission + pruning
    """

    SAVE_PATH = "episteme_data"
    MIN_CONFIDENCE = 0.30

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print("[KG] Initializing Knowledge Graph...", flush=True)

        self.graph = nx.DiGraph()
        self.units: Dict[str, EpistemicUnit] = {}

        # Lazy-load heavy deps to keep startup fast
        self._embedder = None
        self._embedding_model_name = embedding_model
        self.embedding_dim = 384

        self._faiss_index = None
        self.faiss_id_map: List[str] = []

        self.domain_index: Dict[str, List[str]] = {}
        self.total_stored = 0
        self.total_pruned = 0

        os.makedirs(self.SAVE_PATH, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print("[KG] Loading embedding model (first use)...", flush=True)
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    @property
    def faiss_index(self):
        if self._faiss_index is None:
            import faiss as _faiss
            self._faiss_index = _faiss.IndexFlatL2(self.embedding_dim)
        return self._faiss_index

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, eu: EpistemicUnit) -> bool:
        if eu.confidence < self.MIN_CONFIDENCE:
            return False

        # Deduplicate by semantic similarity
        if self.units:
            similar = self.find_similar(eu.proposition, top_k=1, threshold=0.95)
            if similar:
                self._merge(eu, similar[0])
                return True

        # Compute embedding
        eu.embedding = self.embedder.encode(eu.proposition).tolist()

        # Graph
        self.graph.add_node(eu.id, data=eu)
        self.units[eu.id] = eu

        # FAISS
        emb_np = np.array([eu.embedding], dtype=np.float32)
        self.faiss_index.add(emb_np)
        self.faiss_id_map.append(eu.id)

        # Domain index
        self.domain_index.setdefault(eu.domain, []).append(eu.id)

        # Justification edges
        for j_id in eu.justification:
            if j_id in self.units:
                self.graph.add_edge(j_id, eu.id, relation=RelationType.SUPPORTS.value)

        # Relation edges
        for rel_type, rel_ids in eu.relations.items():
            for r_id in rel_ids:
                if r_id in self.units:
                    self.graph.add_edge(eu.id, r_id, relation=rel_type)

        self.total_stored += 1
        return True

    def _merge(self, new_eu: EpistemicUnit, existing: EpistemicUnit):
        if new_eu.confidence > existing.confidence:
            existing.confidence = (existing.confidence + new_eu.confidence) / 2
        existing_names = {s.name for s in existing.sources}
        for s in new_eu.sources:
            if s.name not in existing_names:
                existing.sources.append(s)
        existing.keywords = list(set(existing.keywords + new_eu.keywords))

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.60,
        domain: Optional[str] = None,
    ) -> List[EpistemicUnit]:
        if not self.faiss_id_map:
            return []

        q_emb = self.embedder.encode(query)
        q_np = np.array([q_emb], dtype=np.float32)

        k = min(top_k * 3, len(self.faiss_id_map))
        distances, indices = self.faiss_index.search(q_np, k)

        results: List[Tuple[float, EpistemicUnit]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_id_map):
                continue
            similarity = 1.0 / (1.0 + float(dist))
            if similarity < threshold:
                continue
            eu_id = self.faiss_id_map[idx]
            if eu_id not in self.units:
                continue
            eu = self.units[eu_id]
            if domain and eu.domain != domain:
                continue
            results.append((similarity, eu))

        results.sort(key=lambda x: x[0], reverse=True)
        return [eu for _, eu in results[:top_k]]

    def get_justification_chain(self, eu_id: str, depth: int = 5) -> List[EpistemicUnit]:
        if eu_id not in self.units:
            return []
        chain: List[EpistemicUnit] = []
        visited: set = set()

        def traverse(cid: str, d: int):
            if d == 0 or cid in visited:
                return
            visited.add(cid)
            if cid not in self.units:
                return
            chain.append(self.units[cid])
            for pred in self.graph.predecessors(cid):
                traverse(pred, d - 1)

        traverse(eu_id, depth)
        return chain

    def get_dependent_beliefs(self, eu_id: str) -> List[EpistemicUnit]:
        if eu_id not in self.graph:
            return []
        return [
            self.units[sid]
            for sid in nx.descendants(self.graph, eu_id)
            if sid in self.units
        ]

    def prune(self, threshold: float = 0.30):
        to_remove = [uid for uid, eu in self.units.items() if eu.confidence < threshold]
        for uid in to_remove:
            self.graph.remove_node(uid)
            del self.units[uid]
            self.total_pruned += 1
        if to_remove:
            self._rebuild_faiss()

    def _rebuild_faiss(self):
        import faiss as _faiss
        self._faiss_index = _faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_id_map = []
        for eu in self.units.values():
            if eu.embedding:
                self._faiss_index.add(np.array([eu.embedding], dtype=np.float32))
                self.faiss_id_map.append(eu.id)

    def stats(self) -> Dict:
        confidences = [eu.confidence for eu in self.units.values()]
        return {
            "total_units": len(self.units),
            "total_edges": self.graph.number_of_edges(),
            "total_stored": self.total_stored,
            "total_pruned": self.total_pruned,
            "domains": {d: len(ids) for d, ids in self.domain_index.items()},
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = None):
        path = path or self.SAVE_PATH
        os.makedirs(path, exist_ok=True)

        # Use indent=None for faster write (Kaggle); indent=2 bloats 10k+ units
        units_data = {uid: eu.to_dict() for uid, eu in self.units.items()}
        with open(f"{path}/units.json", "w") as f:
            json.dump(units_data, f, indent=None, separators=(",", ":"))

        with open(f"{path}/graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

        import faiss as _faiss
        _faiss.write_index(self.faiss_index, f"{path}/faiss.index")

        with open(f"{path}/faiss_map.json", "w") as f:
            json.dump(self.faiss_id_map, f)

        print(f"[KG] Saved {len(self.units)} units to {path}/", flush=True)

    def load(self, path: str = None) -> bool:
        path = path or self.SAVE_PATH
        if not os.path.exists(f"{path}/units.json"):
            print("[KG] No saved graph found.", flush=True)
            return False

        with open(f"{path}/units.json", "r") as f:
            units_data = json.load(f)
        self.units = {uid: EpistemicUnit.from_dict(d) for uid, d in units_data.items()}

        with open(f"{path}/graph.pkl", "rb") as f:
            self.graph = pickle.load(f)

        import faiss as _faiss
        self._faiss_index = _faiss.read_index(f"{path}/faiss.index")

        with open(f"{path}/faiss_map.json", "r") as f:
            self.faiss_id_map = json.load(f)

        self.domain_index = {}
        for uid, eu in self.units.items():
            self.domain_index.setdefault(eu.domain, []).append(uid)

        print(f"[KG] Loaded {len(self.units)} units from {path}/", flush=True)
        return True
