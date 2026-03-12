from typing import List
from epichat.core.knowledge_graph import KnowledgeGraph
from .web_seeder import WebSeeder
from .code_seeder import CodeSeeder, CS_WIKI_TOPICS


DEFAULT_TOPICS = CS_WIKI_TOPICS  # code-focused seeding is now the default


class SeedPipeline:
    """Orchestrates the full knowledge seeding process (code-focused)."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg   = knowledge_graph
        self.web  = WebSeeder(knowledge_graph)
        self.code = CodeSeeder(knowledge_graph)

    def run(
        self,
        topics: List[str] = None,
        max_sentences: int = 30,
        include_web: bool = True,
        save_after: bool = True,
    ) -> int:
        total = 0

        # 1. General axioms (logic, physics, epistemology)
        print("\n=== Phase 1: General axioms ===")
        total += self.web.seed_axioms()

        # 2. Curated code knowledge (best practices, patterns, algorithms, Kotlin, security…)
        print("\n=== Phase 2: Code axioms & best practices ===")
        total += self.code.seed_code_axioms()

        # 3. Wikipedia CS articles
        print("\n=== Phase 3: Wikipedia CS articles ===")
        total += self.code.seed_wikipedia(
            topics=topics or DEFAULT_TOPICS,
            max_sentences=max_sentences,
        )

        # 4. Trusted web sources
        if include_web:
            print("\n=== Phase 4: Trusted web sources ===")
            total += self.code.seed_web_sources()

        print(f"\n=== Seeding complete — {total} EUs created ===")
        print(self.kg.stats())

        if save_after:
            self.kg.save()

        return total
