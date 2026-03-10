"""
EpiChat-backed RAG for CodingSSM inference.

Replaces the ChromaDB RAG with EpiChat's epistemically-justified knowledge graph.
Provides structured, confidence-weighted context to the model at inference time.

Key advantages over plain ChromaDB:
- Every retrieved fact carries explicit confidence (0–1)
- Justification chains let us explain WHY we're including a fact
- Domain-scoped retrieval (algorithms, patterns, security, etc.)
- Known epistemic gaps surfaced to the model ("I know I don't know X")
- EUs carry complexity info, tradeoffs, when-to-use — structured, not raw text

Usage:
    from ssm.epichat_rag import EpiChatRAG

    rag = EpiChatRAG("/home/me/EpiChat")
    context = rag.get_context("implement binary search tree")
    # Returns formatted string ready to prepend to model prompt
"""

import json
from pathlib import Path
from typing import Optional


class EpiChatRAG:
    """
    RAG engine backed by EpiChat's knowledge graph + FAISS index.
    Loads lazily on first query.
    """

    # Default: local copy inside SSM project (symlinked from /home/me/EpiChat)
    _DEFAULT_DIR = Path(__file__).parent.parent / "epichat"

    def __init__(
        self,
        epichat_dir: str = None,
        top_k: int = 5,
        min_confidence: float = 0.50,
        max_context_chars: int = 2000,
    ):
        self._dir = Path(epichat_dir) if epichat_dir else self._DEFAULT_DIR
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.max_context_chars = max_context_chars

        self._units: dict = {}
        self._index = None
        self._id_map: list = []
        self._embedder = None
        self._ready = False

    def _load(self):
        if self._ready:
            return
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            units_path = self._dir / "episteme_data" / "units.json"
            faiss_path = self._dir / "episteme_data" / "faiss.index"
            map_path   = self._dir / "episteme_data" / "faiss_map.json"

            with open(units_path) as f:
                data = json.load(f)
            self._units = data if isinstance(data, dict) else {u["id"]: u for u in data}

            self._index = faiss.read_index(str(faiss_path))

            with open(map_path) as f:
                self._id_map = json.load(f)

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._ready = True
            print(f"[EpiChatRAG] Ready — {len(self._units)} knowledge units loaded")
        except Exception as e:
            print(f"[EpiChatRAG] Failed to load: {e}")
            self._ready = True  # prevent infinite retry

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Return top-k EpistemicUnits most relevant to query."""
        self._load()
        if self._index is None:
            return []

        k = top_k or self.top_k
        try:
            import numpy as np
            vec = self._embedder.encode([query], normalize_embeddings=False).astype("float32")
            D, I = self._index.search(vec, k * 4)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                eu_id = self._id_map[idx]
                eu = self._units.get(str(eu_id)) or self._units.get(eu_id)
                if eu is None:
                    continue
                if eu.get("confidence", 0) < self.min_confidence:
                    continue
                sim = 1.0 / (1.0 + float(dist))
                results.append((sim, eu))
            results.sort(key=lambda x: -x[0])
            return [eu for _, eu in results[:k]]
        except Exception as e:
            print(f"[EpiChatRAG] search error: {e}")
            return []

    def search_by_domain(self, query: str, domain: str, top_k: int = 3) -> list[dict]:
        """Return EUs filtered to a specific domain."""
        results = self.search(query, top_k=top_k * 4)
        filtered = [eu for eu in results if eu.get("domain") == domain]
        return filtered[:top_k]

    def get_context(self, query: str, domain: Optional[str] = None) -> str:
        """
        Main entry point: return a formatted context string to prepend to model input.

        Format:
            [Knowledge Context]
            • Binary Search: efficient O(log n) lookup ... [conf: 0.92]
            ...
        """
        if domain:
            eus = self.search_by_domain(query, domain, top_k=self.top_k)
        else:
            eus = self.search(query)

        if not eus:
            return ""

        return self._format(eus)

    def get_complexity_info(self, algorithm_name: str) -> Optional[str]:
        """Targeted lookup: get complexity info for a named algorithm."""
        eus = self.search(algorithm_name, top_k=10)
        for eu in eus:
            tc = eu.get("time_complexity", "")
            sc = eu.get("space_complexity", "")
            if tc or sc:
                prop = eu.get("proposition", algorithm_name)
                result = f"{prop}"
                if tc:
                    result += f"\nTime: {tc}"
                if sc:
                    result += f"\nSpace: {sc}"
                return result
        return None

    def get_design_patterns(self, problem_description: str) -> list[dict]:
        """Find relevant design patterns for a given problem."""
        return self.search_by_domain(problem_description, "design_patterns")

    def get_epistemic_gaps(self, query: str) -> list[str]:
        """Return known unknowns for a topic — useful for model uncertainty awareness."""
        eus = self.search(query, top_k=5)
        gaps = []
        for eu in eus:
            gaps.extend(eu.get("epistemic_gaps", []))
        return list(set(gaps))[:5]

    def _format(self, eus: list[dict]) -> str:
        """Format EUs into a compact, model-readable context block."""
        lines = ["[Knowledge Context]"]
        total_chars = 0

        def _s(v):
            return (v or "").strip()

        for eu in eus:
            prop = _s(eu.get("proposition"))
            if not prop:
                continue
            conf = eu.get("confidence") or 0.0
            domain = _s(eu.get("domain"))
            tc = _s(eu.get("time_complexity"))
            sc = _s(eu.get("space_complexity"))
            when = _s(eu.get("when_to_use"))
            trade = _s(eu.get("tradeoffs"))
            code = _s(eu.get("code_snippet"))
            gaps = eu.get("epistemic_gaps") or []

            entry_parts = [f"• {prop} [conf:{conf:.2f}, domain:{domain}]"]
            if tc:
                entry_parts.append(f"  Time: {tc}")
            if sc:
                entry_parts.append(f"  Space: {sc}")
            if when:
                entry_parts.append(f"  Use when: {when}")
            if trade:
                entry_parts.append(f"  Tradeoffs: {trade}")
            if code and len(code) < 200:
                entry_parts.append(f"  Example: ```python\n{code}\n  ```")
            if gaps:
                entry_parts.append(f"  Known gaps: {'; '.join(gaps[:2])}")

            entry = "\n".join(entry_parts)

            if total_chars + len(entry) > self.max_context_chars:
                break

            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)

    def index_new_knowledge(self, proposition: str, domain: str = "general",
                             confidence: float = 0.7, code_snippet: str = "") -> bool:
        """
        Add new knowledge to EpiChat from model-generated insights.
        Runs if EpiChat is available; no-op otherwise.

        This closes the loop: model generates insights → EpiChat stores them →
        future model queries get richer context.
        """
        self._load()
        if not self._ready or self._index is None:
            return False
        try:
            from epichat.core.knowledge_graph import KnowledgeGraph
            from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType

            kg = KnowledgeGraph(str(self._dir / "episteme_data"))
            eu = EpistemicUnit(
                proposition=proposition,
                knowledge_type=KnowledgeType.INFERRED,
                confidence=confidence,
                domain=domain,
                code_snippet=code_snippet,
            )
            added = kg.add(eu)
            if added:
                kg.save()
            return added
        except Exception as e:
            print(f"[EpiChatRAG] index_new_knowledge failed: {e}")
            return False

    @property
    def stats(self) -> dict:
        self._load()
        return {
            "units": len(self._units),
            "ready": self._ready,
            "index_vectors": self._index.ntotal if self._index else 0,
        }
