import numpy as np
from typing import List
from .epistemic_unit import EpistemicUnit


class BayesianEngine:
    """Confidence calculation and belief revision via Bayesian principles."""

    def calculate_chain_confidence(self, chain: List[EpistemicUnit]) -> float:
        if not chain:
            return 0.0

        # Weakest-link product
        chain_product = float(np.prod([eu.confidence for eu in chain]))

        # Weight by average source reliability
        weights = []
        for eu in chain:
            if eu.sources:
                weights.append(np.mean([s.reliability_score for s in eu.sources]))
            else:
                weights.append(0.5)
        avg_weight = float(np.mean(weights)) if weights else 0.5

        # Each hop introduces a small penalty
        hop_penalty = 0.95 ** len(chain)

        return float(np.clip(chain_product * avg_weight * hop_penalty, 0.0, 1.0))

    def update(
        self,
        prior: float,
        evidence_strength: float,
        evidence_reliability: float = 0.8,
    ) -> float:
        """P(H|E) via Bayes' theorem (simplified)."""
        p_e_given_h = evidence_reliability
        p_e_given_not_h = 1.0 - evidence_reliability
        p_e = p_e_given_h * prior + p_e_given_not_h * (1.0 - prior)
        if p_e == 0:
            return prior
        posterior = p_e_given_h * prior * evidence_strength / p_e
        return float(np.clip(posterior, 0.0, 1.0))

    def downdate(self, prior: float, contradiction_strength: float) -> float:
        return float(np.clip(prior * (1.0 - contradiction_strength * 0.5), 0.0, 1.0))

    def coherence_score(self, eu: EpistemicUnit, related: List[EpistemicUnit]) -> float:
        if not related:
            return 0.5
        supports = [
            r.confidence
            for r in related
            if eu.id in r.relations.get("SUPPORTS", [])
        ]
        contradicts = [
            r.confidence
            for r in related
            if eu.id in r.relations.get("CONTRADICTS", [])
        ]
        s = float(np.mean(supports)) if supports else 0.0
        c = float(np.mean(contradicts)) if contradicts else 0.0
        return float(np.clip(0.5 + (s - c) * 0.5, 0.0, 1.0))
