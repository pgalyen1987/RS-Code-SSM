from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from core.epistemic_unit import EpistemicUnit, KnowledgeType
from core.knowledge_graph import KnowledgeGraph
from core.justifier import Justifier, Judgment
from core.bayesian import BayesianEngine


@dataclass
class QueryResult:
    query: str
    answer: str
    confidence: float
    knowledge_type: str
    justification_chain: List[EpistemicUnit]
    gaps: List[str]
    sources: List[str]
    competing_views: List[EpistemicUnit] = field(default_factory=list)
    reasoning: str = ""

    def is_certain(self) -> bool:
        return self.confidence >= 0.80

    def is_uncertain(self) -> bool:
        return self.confidence < 0.50

    def summary(self) -> str:
        lines = [
            f"Q: {self.query}",
            f"A: {self.answer}",
            f"Confidence: {self.confidence:.1%}  ({self.knowledge_type})",
        ]
        if self.justification_chain:
            lines.append(f"Justified by {len(self.justification_chain)} belief(s)")
        if self.gaps:
            lines.append(f"Gaps ({len(self.gaps)}): {self.gaps[0]}"
                         + (" [...]" if len(self.gaps) > 1 else ""))
        if self.sources:
            lines.append(f"Sources: {', '.join(self.sources[:3])}")
        return "\n".join(lines)


class QueryEngine:
    """
    Translates natural-language questions into epistemic queries against
    the KnowledgeGraph, returning justified answers with confidence scores.
    """

    CERTAINTY_THRESHOLD  = 0.80
    UNCERTAIN_THRESHOLD  = 0.50
    COMPETING_THRESHOLD  = 0.10  # Retrieve competing views within this delta

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg       = knowledge_graph
        self.justifier = Justifier(knowledge_graph)
        self.bayesian  = BayesianEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """Main entry point: answer a natural-language question."""

        # 1. Find relevant EUs
        candidates = self.kg.find_similar(question, top_k=top_k, threshold=0.35)

        if not candidates:
            return self._unknown_result(question)

        # 2. Re-rank by confidence × relevance
        ranked = sorted(candidates, key=lambda eu: eu.confidence, reverse=True)
        best = ranked[0]

        # 3. Evaluate justification
        judgment = self.justifier.evaluate(best)

        # 4. Compose answer
        answer = self._compose_answer(question, best, judgment)

        # 5. Competing views (close confidence, different proposition)
        competing = [
            eu for eu in ranked[1:]
            if abs(eu.confidence - best.confidence) <= self.COMPETING_THRESHOLD
            and eu.id != best.id
        ]

        # 6. Collect unique source names
        sources = list({
            s.name
            for eu in [best] + judgment.chain
            for s in eu.sources
        })

        return QueryResult(
            query=question,
            answer=answer,
            confidence=judgment.confidence,
            knowledge_type=best.knowledge_type.value,
            justification_chain=judgment.chain,
            gaps=judgment.gaps,
            sources=sources,
            competing_views=competing,
            reasoning=judgment.reasoning,
        )

    def ask_why(self, question: str) -> QueryResult:
        """
        Asks for a causal/explanatory chain rather than a direct answer.
        Traverses CAUSES / DEPENDS_ON edges.
        """
        result = self.query(question)
        if not result.justification_chain:
            return result

        # Extend with causal predecessors
        causal_chain: List[EpistemicUnit] = []
        for eu in result.justification_chain[:3]:
            causal_chain.extend(self.kg.get_justification_chain(eu.id, depth=3))

        result.justification_chain = causal_chain
        result.reasoning = (
            "Causal explanation trace:\n" + result.reasoning
        )
        return result

    def what_dont_i_know(self, topic: str) -> List[str]:
        """
        Returns the aggregated epistemic gaps for a topic.
        Useful for meta-cognition — knowing the limits of knowledge.
        """
        candidates = self.kg.find_similar(topic, top_k=10, threshold=0.40)
        gaps: List[str] = []
        seen: set = set()
        for eu in candidates:
            for gap in eu.epistemic_gaps:
                if gap not in seen:
                    gaps.append(gap)
                    seen.add(gap)
        return gaps

    def assert_belief(
        self,
        proposition: str,
        knowledge_type: KnowledgeType = KnowledgeType.EMPIRICAL,
        confidence: float = 0.70,
        domain: str = "general",
        source_name: str = "User",
    ) -> Judgment:
        """
        User or external process asserts a new belief.
        Evaluated and stored if justified.
        """
        from core.epistemic_unit import Source
        eu = EpistemicUnit(
            proposition=proposition,
            knowledge_type=knowledge_type,
            confidence=confidence,
            domain=domain,
            sources=[Source(name=source_name, reliability_score=0.70)],
        )
        judgment = self.justifier.evaluate(eu)
        if judgment.accepted:
            self.kg.add(eu)
        return judgment

    def revise_belief(self, proposition: str, new_evidence: str, evidence_strength: float = 0.6):
        """
        Revise confidence in beliefs matching a proposition,
        given new contradicting or supporting evidence.
        """
        matching = self.kg.find_similar(proposition, top_k=5, threshold=0.50)
        for eu in matching:
            old_conf = eu.confidence
            if evidence_strength > 0:  # supporting
                new_conf = self.bayesian.update(old_conf, evidence_strength)
            else:  # contradicting
                new_conf = self.bayesian.downdate(old_conf, abs(evidence_strength))
            eu.revise_confidence(new_conf, reason=new_evidence)
            # Propagate to dependents
            for dep in self.kg.get_dependent_beliefs(eu.id):
                dep_new = self.bayesian.downdate(dep.confidence, 0.1)
                dep.revise_confidence(dep_new, reason=f"Upstream revision of: {eu.proposition[:40]}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compose_answer(
        self, question: str, eu: EpistemicUnit, judgment: Judgment
    ) -> str:
        if judgment.confidence >= self.CERTAINTY_THRESHOLD:
            return eu.proposition
        if judgment.confidence >= self.UNCERTAIN_THRESHOLD:
            return f"[Moderately confident] {eu.proposition}"
        if eu.knowledge_type == KnowledgeType.HYPOTHETICAL:
            return f"[Hypothetical / unverified] {eu.proposition}"
        return f"[Low confidence — treat with caution] {eu.proposition}"

    # ------------------------------------------------------------------
    # Code-specific query methods
    # ------------------------------------------------------------------

    def explain_concept(self, concept: str) -> QueryResult:
        """
        Explain a programming concept with its justification chain,
        complexity info, and code examples if available.
        """
        candidates = self.kg.find_similar(concept, top_k=8, threshold=0.35)
        if not candidates:
            return self._unknown_result(concept)

        # Prefer EUs with code snippets
        with_code = [eu for eu in candidates if eu.code_snippet]
        ranked = (with_code + [eu for eu in candidates if not eu.code_snippet])
        best = sorted(ranked, key=lambda eu: eu.confidence, reverse=True)[0]

        judgment = self.justifier.evaluate(best)
        sources = list({s.name for eu in [best] + judgment.chain for s in eu.sources})

        return QueryResult(
            query=concept,
            answer=best.proposition,
            confidence=judgment.confidence,
            knowledge_type=best.knowledge_type.value,
            justification_chain=judgment.chain,
            gaps=judgment.gaps,
            sources=sources,
            competing_views=[eu for eu in candidates[1:4] if eu.id != best.id],
            reasoning=judgment.reasoning,
        )

    def find_pattern(self, problem: str) -> List[EpistemicUnit]:
        """
        Return design patterns or algorithms relevant to a described problem.
        """
        candidates = self.kg.find_similar(problem, top_k=10, threshold=0.35)
        pattern_types = {
            KnowledgeType.DESIGN_PATTERN,
            KnowledgeType.ALGORITHM,
            KnowledgeType.BEST_PRACTICE,
        }
        return [eu for eu in candidates if eu.knowledge_type in pattern_types]

    def get_complexity(self, algorithm: str) -> Optional[EpistemicUnit]:
        """Return an EU with complexity info for the named algorithm."""
        candidates = self.kg.find_similar(algorithm, top_k=10, threshold=0.40)
        for eu in candidates:
            if eu.time_complexity or eu.space_complexity:
                return eu
        return None

    def find_tradeoffs(self, topic: str) -> List[EpistemicUnit]:
        """Return EUs describing tradeoffs for a given topic."""
        candidates = self.kg.find_similar(topic, top_k=10, threshold=0.35)
        return [
            eu for eu in candidates
            if eu.tradeoffs or eu.knowledge_type == KnowledgeType.TRADEOFF
        ]

    def find_code_examples(self, topic: str, language: str = None) -> List[EpistemicUnit]:
        """Return EUs that have code snippets for the given topic."""
        candidates = self.kg.find_similar(topic, top_k=15, threshold=0.35)
        with_code = [eu for eu in candidates if eu.code_snippet]
        if language:
            lang_lower = language.lower()
            with_code = [
                eu for eu in with_code
                if eu.language and eu.language.lower() == lang_lower
            ]
        return with_code

    def security_check(self, topic: str) -> List[EpistemicUnit]:
        """Return security-relevant EUs for a given programming topic."""
        candidates = self.kg.find_similar(topic, top_k=10, threshold=0.35)
        return [eu for eu in candidates if eu.domain == "cybersecurity"]

    # ------------------------------------------------------------------

    def _unknown_result(self, question: str) -> QueryResult:
        return QueryResult(
            query=question,
            answer="No relevant knowledge found in the epistemic graph.",
            confidence=0.0,
            knowledge_type="UNKNOWN",
            justification_chain=[],
            gaps=["Topic not yet seeded into the knowledge graph"],
            sources=[],
        )
