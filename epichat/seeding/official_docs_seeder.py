"""
Seed EpistemicUnits from official vendor / standards documentation (HTML fetch).
"""
from __future__ import annotations

import re
import time
from typing import List

import requests
from bs4 import BeautifulSoup

from epichat.core.epistemic_unit import EpistemicUnit, KnowledgeType, Source
from epichat.core.knowledge_graph import KnowledgeGraph

from .official_docs_registry import OFFICIAL_DOCUMENTATION_SOURCES


_DEFAULT_UA = (
    "Mozilla/5.0 (compatible; EPISTEME-docs-seeder/1.0; "
    "+https://github.com/pgalyen1987/RS-Code-SSM) AppleWebKit/537.36"
)


def _clean_text(raw: str) -> str:
    t = re.sub(r"\s+", " ", raw).strip()
    t = re.sub(r"\[\s*\d+\s*\]", "", t)
    return t.strip()


def _paragraphs_from_soup(soup: BeautifulSoup, max_paragraphs: int) -> List[str]:
    candidates: List[str] = []
    selectors = (
        "article p",
        "main p",
        "div[role='main'] p",
        "#content p",
        ".content p",
        "div.content p",
        ".markdown p",
        "#main-col-body p",
        ".mw-parser-output p",
    )
    seen = set()
    for sel in selectors:
        for p in soup.select(sel):
            t = _clean_text(p.get_text())
            if len(t) < 60 or t in seen:
                continue
            seen.add(t)
            candidates.append(t)
            if len(candidates) >= max_paragraphs:
                return candidates
    if len(candidates) < max_paragraphs // 4:
        for p in soup.find_all("p"):
            t = _clean_text(p.get_text())
            if len(t) < 60 or t in seen:
                continue
            seen.add(t)
            candidates.append(t)
            if len(candidates) >= max_paragraphs:
                break
    return candidates[:max_paragraphs]


class OfficialDocsSeeder:
    """Fetch official docs HTML and add paragraph-sized EpistemicUnits."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def seed(
        self,
        sources=None,
        max_paragraphs_per_url: int = 40,
        delay_sec: float = 0.4,
        proposition_max_len: int = 520,
    ) -> int:
        sources = sources or OFFICIAL_DOCUMENTATION_SOURCES
        total = 0
        headers = {"User-Agent": _DEFAULT_UA, "Accept-Language": "en-US,en;q=0.9"}

        for src in sources:
            url = src["url"]
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[OfficialDocs] skip {url}: {e}", flush=True)
                time.sleep(delay_sec)
                continue

            ctype = (resp.headers.get("content-type") or "").lower()
            if "html" not in ctype and not url.endswith(".html"):
                print(f"[OfficialDocs] skip non-html {url}", flush=True)
                time.sleep(delay_sec)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            paras = _paragraphs_from_soup(soup, max_paragraphs_per_url)
            rel = float(src.get("reliability", 0.92))
            domain = src["domain"]
            lang = src.get("language")
            name = src["name"]

            for para in paras:
                prop = para[:proposition_max_len]
                eu = EpistemicUnit(
                    proposition=prop,
                    knowledge_type=KnowledgeType.EMPIRICAL,
                    confidence=min(0.94, rel * 0.88),
                    domain=domain,
                    sources=[
                        Source(
                            name=name,
                            url=url,
                            reliability_score=rel,
                        )
                    ],
                    keywords=[x for x in (domain, (lang or "").lower()) if x],
                    language=lang,
                )
                if self.kg.add(eu):
                    total += 1

            print(f"[OfficialDocs] {name}: +{len(paras)} paragraphs scraped", flush=True)
            time.sleep(delay_sec)

        print(f"[OfficialDocs] Total new EUs admitted: {total}", flush=True)
        return total
