import re
import nltk
from typing import List
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from epichat.core.knowledge_graph import KnowledgeGraph
from epichat.core.justifier import Justifier


class WikipediaSeeder:
    """Converts Wikipedia articles into justified EpistemicUnits."""

    WIKI_RELIABILITY = 0.82
    BASE_CONFIDENCE  = 0.75

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.justifier = Justifier(knowledge_graph)
        self._stop_words = set(stopwords.words("english"))
        self._seeded: set = set()

        import wikipediaapi
        self.wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="EPISTEME/1.0 (knowledge-seeder)",
        )

    # ------------------------------------------------------------------

    def seed_topic(self, topic: str, depth: int = 1, max_sentences: int = 50) -> int:
        if topic in self._seeded:
            return 0
        self._seeded.add(topic)

        print(f"[Wiki] Seeding: {topic}")
        page = self.wiki.page(topic)
        if not page.exists():
            print(f"[Wiki] Page not found: {topic}")
            return 0

        domain = self._infer_domain(page)

        # Anchor EU for the topic
        anchor = EpistemicUnit(
            proposition=f"{page.title}: {page.summary[:200]}",
            knowledge_type=KnowledgeType.EMPIRICAL,
            confidence=self.BASE_CONFIDENCE,
            domain=domain,
            sources=[Source(name="Wikipedia", url=page.fullurl,
                            reliability_score=self.WIKI_RELIABILITY)],
            keywords=self._keywords(page.title + " " + page.summary),
        )
        self.kg.add(anchor)
        created = 1

        sentences = self._extract_sentences(page.text, max_sentences)

        for sent in tqdm(sentences, desc=f"  {topic}", leave=False):
            eu = EpistemicUnit(
                proposition=sent,
                knowledge_type=KnowledgeType.EMPIRICAL,
                confidence=self._sentence_confidence(sent),
                domain=domain,
                justification=[anchor.id],
                sources=[Source(name=f"Wikipedia: {page.title}",
                                url=page.fullurl,
                                reliability_score=self.WIKI_RELIABILITY)],
                keywords=self._keywords(sent),
                epistemic_gaps=self._sentence_gaps(sent),
            )
            judgment = self.justifier.evaluate(eu)
            if judgment.accepted:
                eu.confidence = min(eu.confidence, judgment.confidence + 0.1)
                if self.kg.add(eu):
                    created += 1

        # Recurse into linked pages
        if depth > 1 and created > 0:
            for linked in list(page.links.keys())[:5]:
                created += self.seed_topic(linked, depth=depth - 1, max_sentences=15)

        print(f"[Wiki] {topic}: +{created} EUs")
        return created

    def seed_batch(self, topics: List[str], depth: int = 1) -> int:
        total = sum(self.seed_topic(t, depth=depth) for t in topics)
        print(f"[Wiki] Total EUs: {total}")
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_sentences(self, text: str, max_n: int) -> List[str]:
        text = re.sub(r"\[\[.*?\]\]", "", text)
        text = re.sub(r"\{\{.*?\}\}", "", text)
        text = re.sub(r"={2,}.*?={2,}", "", text)
        text = re.sub(r"\s+", " ", text)

        sents = []
        for s in sent_tokenize(text):
            s = s.strip()
            if 30 < len(s) < 500 and not s.startswith(("*", "#", "|", "!")):
                if any(w in s.lower() for w in
                       ["is", "are", "was", "were", "has", "have", "had",
                        "will", "would", "can", "could"]):
                    sents.append(s)
            if len(sents) >= max_n:
                break
        return sents

    def _keywords(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return list({w for w in words if w not in self._stop_words})[:10]

    def _infer_domain(self, page) -> str:
        mapping = {
            "physics": "physics", "chemistry": "chemistry",
            "biology": "biology", "mathematics": "mathematics",
            "computer": "computer_science", "history": "history",
            "geography": "geography", "philosophy": "philosophy",
            "medicine": "medicine", "economics": "economics",
            "psychology": "psychology", "astronomy": "astronomy",
        }
        text = (page.title + " " + str(list(page.categories.keys())[:10])).lower()
        for kw, dom in mapping.items():
            if kw in text:
                return dom
        return "general"

    def _sentence_confidence(self, sentence: str) -> float:
        base = 0.75
        hedging  = ["might", "may", "could", "possibly", "perhaps", "probably",
                    "likely", "suggests", "appears", "seems", "approximately",
                    "estimated", "believed", "thought"]
        certainty = ["is", "are", "was", "proven", "demonstrated", "confirmed",
                     "established", "known", "defined"]
        sl = sentence.lower()
        conf = base - sum(0.05 for w in hedging if w in sl) \
                    + sum(0.02 for w in certainty if w in sl)
        return float(max(0.30, min(0.95, conf)))

    def _sentence_gaps(self, sentence: str) -> List[str]:
        gaps = []
        sl = sentence.lower()
        if any(w in sl for w in ["unclear", "unknown", "debated", "disputed"]):
            gaps.append("Explicitly marked as uncertain in source")
        if re.search(r"\d{4}", sentence):
            gaps.append("Contains date claims that may need verification")
        if "according to" in sl:
            gaps.append("Based on a specific source — may not be universally accepted")
        return gaps
