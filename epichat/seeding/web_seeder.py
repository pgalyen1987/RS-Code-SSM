import re
import time
import requests
from bs4 import BeautifulSoup
from typing import List

from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from epichat.core.knowledge_graph import KnowledgeGraph


AXIOMS: List[tuple] = [
    # Logic
    ("A statement cannot be both true and false simultaneously", "logic", 1.0),
    ("If A implies B, and A is true, then B is true (modus ponens)", "logic", 1.0),
    ("Everything is identical to itself (law of identity)", "logic", 1.0),
    # Mathematics
    ("1 + 1 = 2 in standard arithmetic", "mathematics", 1.0),
    ("The whole is greater than any of its parts", "mathematics", 1.0),
    ("Between any two real numbers there exists another real number", "mathematics", 1.0),
    # Physics
    ("Energy cannot be created or destroyed, only transformed", "physics", 0.99),
    ("Objects in motion tend to stay in motion unless acted upon by a net force", "physics", 0.99),
    ("For every action there is an equal and opposite reaction", "physics", 0.99),
    ("The speed of light in a vacuum is constant for all observers", "physics", 0.99),
    # Epistemology
    ("Knowledge requires justification, truth, and belief (Justified True Belief)", "philosophy", 0.85),
    ("Perception can be fallible and requires independent verification", "philosophy", 0.92),
    ("Absence of evidence is not evidence of absence", "philosophy", 0.90),
    ("Simpler explanations are preferable to more complex ones when equally valid (Occam's Razor)", "philosophy", 0.88),
    # Biology
    ("All living organisms are composed of cells", "biology", 0.99),
    ("DNA encodes genetic information in living organisms", "biology", 0.99),
    ("Evolution occurs through natural selection acting on heritable variation", "biology", 0.98),
]


class WebSeeder:
    """Seeds knowledge from trusted web sources and foundational axioms."""

    HEADERS = {"User-Agent": "EPISTEME/1.0 Knowledge Seeder"}

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def seed_axioms(self) -> int:
        created = 0
        for prop, domain, conf in AXIOMS:
            eu = EpistemicUnit(
                proposition=prop,
                knowledge_type=KnowledgeType.AXIOM,
                confidence=conf,
                domain=domain,
                sources=[Source(name="Foundational Axioms", reliability_score=1.0)],
            )
            if self.kg.add(eu):
                created += 1
        print(f"[Axioms] Seeded {created} foundational axioms")
        return created

    def seed_from_url(
        self,
        url: str,
        source_name: str,
        domain: str,
        reliability: float = 0.75,
        max_paragraphs: int = 40,
    ) -> int:
        try:
            print(f"[Web] Fetching: {url}")
            resp = requests.get(url, headers=self.HEADERS, timeout=12)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[Web] Error: {e}")
            return 0

        soup = BeautifulSoup(resp.text, "html.parser")
        created = 0

        for para in soup.find_all("p")[:max_paragraphs]:
            text = re.sub(r"\s+", " ", para.get_text()).strip()
            text = re.sub(r"\[.*?\]", "", text).strip()  # remove citation markers

            if len(text) < 60:
                continue

            eu = EpistemicUnit(
                proposition=text[:500],
                knowledge_type=KnowledgeType.EMPIRICAL,
                confidence=reliability * 0.85,
                domain=domain,
                sources=[Source(name=source_name, url=url,
                                reliability_score=reliability)],
            )
            if self.kg.add(eu):
                created += 1

        time.sleep(0.5)
        print(f"[Web] {source_name}: +{created} EUs")
        return created
