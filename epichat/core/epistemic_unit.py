from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


class KnowledgeType(Enum):
    AXIOM          = "AXIOM"          # Self-evident, a priori
    EMPIRICAL      = "EMPIRICAL"      # Directly observed/sourced
    INFERRED       = "INFERRED"       # Derived from other EUs
    HYPOTHETICAL   = "HYPOTHETICAL"   # Unverified
    # Code-specific types
    CODE_EXAMPLE   = "CODE_EXAMPLE"   # Verified code snippet
    ALGORITHM      = "ALGORITHM"      # Named algorithm with complexity
    DESIGN_PATTERN = "DESIGN_PATTERN" # Architectural/GoF pattern
    BEST_PRACTICE  = "BEST_PRACTICE"  # Community-accepted guideline
    TRADEOFF       = "TRADEOFF"       # When/why to choose X over Y


class RelationType(Enum):
    SUPPORTS    = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DEPENDS_ON  = "DEPENDS_ON"
    GENERALIZES = "GENERALIZES"
    SPECIALIZES = "SPECIALIZES"
    CAUSES      = "CAUSES"
    CAUSED_BY   = "CAUSED_BY"


@dataclass
class Source:
    name: str
    url: Optional[str] = None
    reliability_score: float = 0.7
    retrieved_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "url": self.url,
            "reliability_score": self.reliability_score,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass
class BeliefRevision:
    reason: str
    old_confidence: float
    new_confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    triggered_by: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "reason": self.reason,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "timestamp": self.timestamp.isoformat(),
            "triggered_by": self.triggered_by,
        }


@dataclass
class EpistemicUnit:
    """
    Fundamental unit of knowledge in EPISTEME.
    Every piece of knowledge carries its justification chain.
    """
    proposition: str
    knowledge_type: KnowledgeType
    confidence: float

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = "general"

    justification: List[str] = field(default_factory=list)   # EU ids
    sources: List[Source] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    revision_history: List[BeliefRevision] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    epistemic_gaps: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)  # RelationType.value -> [eu_id]

    # Code-specific metadata (None for non-code EUs)
    code_snippet: Optional[str] = None         # Actual code example
    language: Optional[str] = None             # Python, JavaScript, etc.
    time_complexity: Optional[str] = None      # e.g. "O(n log n)"
    space_complexity: Optional[str] = None     # e.g. "O(n)"
    when_to_use: Optional[str] = None          # Guidance on applicability
    tradeoffs: Optional[str] = None            # What you give up

    def add_relation(self, relation_type: RelationType, eu_id: str):
        key = relation_type.value
        if key not in self.relations:
            self.relations[key] = []
        if eu_id not in self.relations[key]:
            self.relations[key].append(eu_id)

    def revise_confidence(self, new_confidence: float, reason: str, triggered_by: str = None):
        self.revision_history.append(BeliefRevision(
            reason=reason,
            old_confidence=self.confidence,
            new_confidence=new_confidence,
            triggered_by=triggered_by,
        ))
        self.confidence = new_confidence

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "knowledge_type": self.knowledge_type.value,
            "confidence": self.confidence,
            "domain": self.domain,
            "justification": self.justification,
            "sources": [s.to_dict() for s in self.sources],
            "keywords": self.keywords,
            "revision_history": [r.to_dict() for r in self.revision_history],
            "created_at": self.created_at.isoformat(),
            "epistemic_gaps": self.epistemic_gaps,
            "relations": self.relations,
            "code_snippet": self.code_snippet,
            "language": self.language,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "when_to_use": self.when_to_use,
            "tradeoffs": self.tradeoffs,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> EpistemicUnit:
        eu = cls(
            proposition=data["proposition"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            confidence=data["confidence"],
            id=data["id"],
            domain=data.get("domain", "general"),
            justification=data.get("justification", []),
            keywords=data.get("keywords", []),
            epistemic_gaps=data.get("epistemic_gaps", []),
            relations=data.get("relations", {}),
        )
        eu.sources = [
            Source(
                name=s["name"],
                url=s.get("url"),
                reliability_score=s.get("reliability_score", 0.7),
            )
            for s in data.get("sources", [])
        ]
        eu.code_snippet    = data.get("code_snippet")
        eu.language        = data.get("language")
        eu.time_complexity = data.get("time_complexity")
        eu.space_complexity= data.get("space_complexity")
        eu.when_to_use     = data.get("when_to_use")
        eu.tradeoffs       = data.get("tradeoffs")
        return eu

    def __repr__(self):
        return (
            f"EU(type={self.knowledge_type.value}, "
            f"conf={self.confidence:.2f}, "
            f"'{self.proposition[:60]}...')"
        )
