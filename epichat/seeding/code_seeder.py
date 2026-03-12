"""
CodeSeeder: seeds the knowledge graph with programming-specific knowledge.
Combines curated code axioms with Wikipedia CS articles and trusted doc pages.
"""
from __future__ import annotations
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import List
from tqdm import tqdm

from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from epichat.core.knowledge_graph import KnowledgeGraph
from epichat.core.justifier import Justifier
from .code_axioms import CODE_KNOWLEDGE


# Wikipedia articles covering CS, OOP, networking, security
CS_WIKI_TOPICS: List[str] = [
    # Algorithms & DS
    "Algorithm", "Data structure", "Big O notation",
    "Sorting algorithm", "Binary search algorithm",
    "Dynamic programming", "Graph theory", "Hash table",
    "Tree (data structure)", "Linked list", "Stack (abstract data type)",
    "Queue (abstract data type)", "Heap (data structure)",
    # OOP
    "Object-oriented programming", "Encapsulation (computer programming)",
    "Inheritance (object-oriented programming)", "Polymorphism (computer science)",
    "SOLID", "Design Patterns (book)", "Software design pattern",
    "Singleton pattern", "Observer pattern", "Factory method pattern",
    "Dependency injection",
    # Languages
    "Python (programming language)", "Kotlin (programming language)",
    "Java (programming language)", "JavaScript", "TypeScript",
    "Functional programming", "Type system",
    # Architecture
    "Software architecture", "Microservices", "Model–view–controller",
    "REST", "Event-driven architecture", "CQRS",
    # Networking
    "OSI model", "Transmission Control Protocol", "User Datagram Protocol",
    "Domain Name System", "Transport Layer Security", "HTTP",
    "WebSocket", "Computer network", "Internet protocol suite",
    "Load balancing (computing)", "Content delivery network",
    # Security
    "Computer security", "OWASP", "SQL injection",
    "Cross-site scripting", "Cross-site request forgery",
    "Public-key cryptography", "Symmetric-key algorithm",
    "Transport Layer Security", "Authentication", "Authorization",
    "Penetration testing", "Firewall (computing)",
    # Testing & practices
    "Test-driven development", "Unit testing", "Code review",
    "Software engineering", "Agile software development",
    "Version control", "Continuous integration",
    "Refactoring", "Technical debt",
]

# Trusted web sources for deep programming knowledge
TRUSTED_CODE_SOURCES: List[dict] = [
    {
        "url": "https://refactoring.guru/design-patterns/catalog",
        "name": "Refactoring Guru: Design Patterns",
        "domain": "design_patterns",
        "reliability": 0.90,
    },
]


class CodeSeeder:
    """Seeds the knowledge graph with code-focused knowledge."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.justifier = Justifier(knowledge_graph)

    # ------------------------------------------------------------------ #
    # Curated code axioms                                                  #
    # ------------------------------------------------------------------ #

    def seed_code_axioms(self) -> int:
        created = 0
        for prop, domain, conf, ktype, extras in tqdm(CODE_KNOWLEDGE, desc="Code axioms"):
            eu = EpistemicUnit(
                proposition=prop,
                knowledge_type=ktype,
                confidence=conf,
                domain=domain,
                sources=[Source(name="EPISTEME Code Knowledge Base", reliability_score=0.95)],
                keywords=extras.get("keywords", []),
                epistemic_gaps=[],
                code_snippet=extras.get("code_snippet"),
                language=extras.get("language"),
                time_complexity=extras.get("time_complexity"),
                space_complexity=extras.get("space_complexity"),
                when_to_use=extras.get("when_to_use"),
                tradeoffs=extras.get("tradeoffs"),
            )
            if self.kg.add(eu):
                created += 1

        print(f"[Code] Seeded {created} code axioms / best-practices / patterns")
        return created

    # ------------------------------------------------------------------ #
    # Wikipedia CS topics                                                  #
    # ------------------------------------------------------------------ #

    def seed_wikipedia(
        self,
        topics: List[str] = None,
        max_sentences: int = 30,
    ) -> int:
        import wikipediaapi
        import nltk
        from nltk.tokenize import sent_tokenize
        from nltk.corpus import stopwords

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

        stop_words = set(stopwords.words("english"))
        topics = topics or CS_WIKI_TOPICS
        seeded: set = set()

        wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="EPISTEME/1.0 code-seeder",
        )

        total = 0
        for topic in tqdm(topics, desc="Wikipedia CS"):
            if topic in seeded:
                continue
            seeded.add(topic)

            page = wiki.page(topic)
            if not page.exists():
                continue

            domain = self._infer_code_domain(page.title)

            anchor = EpistemicUnit(
                proposition=f"{page.title}: {page.summary[:200]}",
                knowledge_type=KnowledgeType.EMPIRICAL,
                confidence=0.78,
                domain=domain,
                sources=[Source(
                    name="Wikipedia",
                    url=page.fullurl,
                    reliability_score=0.82,
                )],
                keywords=self._keywords(page.title + " " + page.summary, stop_words),
            )
            self.kg.add(anchor)
            total += 1

            sentences = self._extract_sentences(page.text, max_sentences)
            for sent in sentences:
                eu = EpistemicUnit(
                    proposition=sent,
                    knowledge_type=KnowledgeType.EMPIRICAL,
                    confidence=self._sentence_confidence(sent),
                    domain=domain,
                    justification=[anchor.id],
                    sources=[Source(
                        name=f"Wikipedia: {page.title}",
                        url=page.fullurl,
                        reliability_score=0.82,
                    )],
                    keywords=self._keywords(sent, stop_words),
                )
                j = self.justifier.evaluate(eu)
                if j.accepted:
                    eu.confidence = min(eu.confidence, j.confidence + 0.1)
                    if self.kg.add(eu):
                        total += 1

        print(f"[Wiki-CS] {total} EUs from Wikipedia CS articles")
        return total

    # ------------------------------------------------------------------ #
    # Trusted web docs                                                     #
    # ------------------------------------------------------------------ #

    def seed_web_sources(self, sources: List[dict] = None) -> int:
        sources = sources or TRUSTED_CODE_SOURCES
        total = 0
        headers = {"User-Agent": "EPISTEME/1.0"}
        for src in sources:
            try:
                resp = requests.get(src["url"], headers=headers, timeout=12)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[Web] Error {src['url']}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for para in soup.find_all("p")[:40]:
                text = re.sub(r"\s+", " ", para.get_text()).strip()
                text = re.sub(r"\[.*?\]", "", text).strip()
                if len(text) < 60:
                    continue
                eu = EpistemicUnit(
                    proposition=text[:500],
                    knowledge_type=KnowledgeType.EMPIRICAL,
                    confidence=src["reliability"] * 0.85,
                    domain=src["domain"],
                    sources=[Source(
                        name=src["name"],
                        url=src["url"],
                        reliability_score=src["reliability"],
                    )],
                )
                if self.kg.add(eu):
                    total += 1
            time.sleep(0.5)

        print(f"[Web-CS] {total} EUs from trusted code sources")
        return total

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _extract_sentences(self, text: str, max_n: int) -> List[str]:
        from nltk.tokenize import sent_tokenize
        text = re.sub(r"\[\[.*?\]\]", "", text)
        text = re.sub(r"\{\{.*?\}\}", "", text)
        text = re.sub(r"={2,}.*?={2,}", "", text)
        text = re.sub(r"\s+", " ", text)

        out = []
        for s in sent_tokenize(text):
            s = s.strip()
            if 30 < len(s) < 500 and not s.startswith(("*", "#", "|", "!")):
                if any(w in s.lower() for w in
                       ["is", "are", "was", "were", "has", "have", "had",
                        "will", "would", "can", "could"]):
                    out.append(s)
            if len(out) >= max_n:
                break
        return out

    def _keywords(self, text: str, stop_words: set) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return list({w for w in words if w not in stop_words})[:10]

    def _infer_code_domain(self, title: str) -> str:
        t = title.lower()
        mapping = {
            ("algorithm", "sort", "search", "dynamic programming",
             "graph", "complexity", "big o"): "algorithms",
            ("data structure", "tree", "heap", "hash", "linked list",
             "stack", "queue", "trie"): "data_structures",
            ("pattern", "singleton", "observer", "factory", "decorator",
             "strategy", "adapter", "facade"): "design_patterns",
            ("object-oriented", "inheritance", "polymorphism",
             "encapsulation", "solid"): "oop",
            ("kotlin",): "kotlin",
            ("python",): "python",
            ("javascript", "typescript"): "javascript",
            ("java ",): "java",
            ("tcp", "udp", "http", "dns", "tls", "socket", "network",
             "osi", "load balan", "cdn", "websocket"): "networking",
            ("security", "xss", "csrf", "sql injection", "owasp",
             "cryptograph", "authentication", "authorization",
             "penetration"): "cybersecurity",
            ("test", "tdd", "unit test"): "testing",
            ("agile", "scrum", "continuous", "devops", "refactor"): "best_practices",
            ("architect", "microservice", "mvc", "rest", "event-driven",
             "cqrs"): "architecture",
        }
        for keywords, domain in mapping.items():
            if any(kw in t for kw in keywords):
                return domain
        return "computer_science"

    def _sentence_confidence(self, sentence: str) -> float:
        base = 0.75
        hedging  = ["might", "may", "could", "possibly", "perhaps", "probably",
                    "likely", "suggests", "appears", "seems", "approximately",
                    "estimated", "believed"]
        certainty = ["is", "are", "was", "proven", "demonstrated", "confirmed",
                     "established", "known", "defined", "always", "never"]
        sl = sentence.lower()
        conf = base - sum(0.05 for w in hedging if w in sl) \
                    + sum(0.02 for w in certainty if w in sl)
        return float(max(0.30, min(0.95, conf)))
