from __future__ import annotations
from typing import List, TYPE_CHECKING
from dataclasses import dataclass

from .epistemic_unit import EpistemicUnit, KnowledgeType
from .bayesian import BayesianEngine

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph


@dataclass
class Judgment:
    accepted: bool
    confidence: float
    gaps: List[str]
    circular_detected: bool
    chain: List[EpistemicUnit]
    reasoning: str


class Justifier:
    """
    The epistemological core of EPISTEME.
    Evaluates whether an EpistemicUnit is justified.
    """

    CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.bayesian = BayesianEngine()

    def evaluate(self, eu: EpistemicUnit) -> Judgment:
        # Axioms are self-justifying
        if eu.knowledge_type == KnowledgeType.AXIOM:
            return Judgment(
                accepted=True,
                confidence=eu.confidence,
                gaps=[],
                circular_detected=False,
                chain=[eu],
                reasoning="Self-evident axiom — no external justification required.",
            )

        chain = self.kg.get_justification_chain(eu.id)
        circular = self._detect_circular(eu, chain)
        consistent = self._check_consistency(eu, chain)

        chain_conf = self.bayesian.calculate_chain_confidence(chain) if chain else eu.confidence

        related = self.kg.find_similar(eu.proposition, top_k=5)
        coherence = self.bayesian.coherence_score(eu, related)

        if eu.knowledge_type == KnowledgeType.EMPIRICAL:
            src_rel = (
                sum(s.reliability_score for s in eu.sources) / len(eu.sources)
                if eu.sources else 0.5
            )
            final_conf = eu.confidence * src_rel
        else:
            final_conf = chain_conf * coherence

        gaps = self._identify_gaps(eu, chain)
        reasoning = self._build_reasoning(eu, chain, circular, consistent)

        accepted = (
            not circular
            and consistent
            and final_conf >= self.CONFIDENCE_THRESHOLD
        )

        return Judgment(
            accepted=accepted,
            confidence=float(final_conf),
            gaps=gaps,
            circular_detected=circular,
            chain=chain,
            reasoning=reasoning,
        )

    def _detect_circular(self, eu: EpistemicUnit, chain: List[EpistemicUnit]) -> bool:
        return eu.id in {c.id for c in chain}

    def _check_consistency(self, eu: EpistemicUnit, chain: List[EpistemicUnit]) -> bool:
        for c in chain:
            if eu.id in c.relations.get("CONTRADICTS", []):
                return False
        return True

    def _identify_gaps(self, eu: EpistemicUnit, chain: List[EpistemicUnit]) -> List[str]:
        gaps = list(eu.epistemic_gaps)
        for c in chain:
            if c.confidence < 0.6:
                gaps.append(
                    f"Uncertain link: '{c.proposition[:50]}' (conf={c.confidence:.2f})"
                )
            if c.knowledge_type == KnowledgeType.HYPOTHETICAL:
                gaps.append(f"Hypothetical assumption: '{c.proposition[:50]}'")
        for j_id in eu.justification:
            if j_id not in self.kg.units:
                gaps.append(f"Missing justification node: {j_id[:8]}...")
        return gaps

    def _build_reasoning(
        self,
        eu: EpistemicUnit,
        chain: List[EpistemicUnit],
        circular: bool,
        consistent: bool,
    ) -> str:
        parts = []
        if circular:
            parts.append("WARNING: Circular reasoning detected.")
        if not consistent:
            parts.append("WARNING: Inconsistency with knowledge base detected.")
        if chain:
            parts.append(f"Justified by {len(chain)} prior belief(s):")
            for i, c in enumerate(chain[:5]):
                connector = "└─" if i == len(chain) - 1 else "├─"
                parts.append(
                    f"  {connector} [{c.knowledge_type.value}|{c.confidence:.2f}] "
                    f"{c.proposition[:70]}"
                )
        if eu.sources:
            parts.append(f"Sources: {', '.join(s.name for s in eu.sources[:3])}")
        return "\n".join(parts)
