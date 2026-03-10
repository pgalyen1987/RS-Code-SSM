"""RAG engine: code chunking, embedding, and retrieval via ChromaDB."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".java", ".rb", ".php", ".swift", ".kt",
    ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".json",
    ".sql", ".r", ".jl", ".lua", ".ex", ".exs",
    ".md", ".txt",
}

IGNORE_DIRS = {
    ".git", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build",
    ".idea", ".vscode", "*.egg-info",
}

# Regex patterns that mark good chunk boundaries per language
_SPLIT_PATTERNS = re.compile(
    r"^(class |def |async def |function |const |let |var |fn |pub fn |impl |struct |enum |interface )",
    re.MULTILINE,
)


def _smart_chunks(text: str, file_path: str, chunk_size: int = 60, overlap: int = 10) -> list[dict]:
    """
    Split code into overlapping chunks, preferring to break on definition
    boundaries (class, def, function, etc.) when possible.
    """
    lines = text.splitlines()
    # Find line indices that are definition boundaries
    boundaries = {0}
    for i, line in enumerate(lines):
        if _SPLIT_PATTERNS.match(line.lstrip()):
            boundaries.add(i)
    boundaries.add(len(lines))
    boundaries = sorted(boundaries)

    # Group boundary spans into chunks of ~chunk_size lines
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))

        # Extend to the next definition boundary if close
        for b in boundaries:
            if b >= end:
                if b - end <= overlap:
                    end = b
                break

        chunk_text = "\n".join(lines[start:end]).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "file": file_path,
                "start_line": start + 1,
            })

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


class RAGEngine:
    """Persistent code search engine backed by ChromaDB + sentence-transformers."""

    def __init__(self, db_path: Optional[str] = None):
        from ssm import config as cfg

        db_path = db_path or cfg.get("rag_db_path")
        db_path = str(Path(db_path).expanduser())

        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None  # lazy-loaded

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def index_directory(
        self,
        directory: str,
        chunk_size: int = 60,
        chunk_overlap: int = 10,
        progress_callback=None,
    ) -> int:
        """
        Walk directory, chunk all code files, embed and upsert into ChromaDB.
        Returns the total number of chunks indexed.
        """
        dir_path = Path(directory).resolve()
        files = [
            f for f in dir_path.rglob("*")
            if f.is_file()
            and f.suffix in CODE_EXTENSIONS
            and not any(part in IGNORE_DIRS for part in f.parts)
        ]

        total_chunks = 0
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(i, len(files), str(file_path.relative_to(dir_path)))

            try:
                text = file_path.read_text(errors="ignore")
                rel_path = str(file_path.relative_to(dir_path))
                chunks = _smart_chunks(text, rel_path, chunk_size, chunk_overlap)

                if not chunks:
                    continue

                texts = [c["text"] for c in chunks]
                embeddings = self._embed(texts)
                ids = [
                    hashlib.md5(f"{rel_path}:{c['start_line']}".encode()).hexdigest()
                    for c in chunks
                ]
                metadatas = [
                    {"file": c["file"], "start_line": c["start_line"]}
                    for c in chunks
                ]

                BATCH = 100
                for j in range(0, len(ids), BATCH):
                    self.collection.upsert(
                        ids=ids[j:j + BATCH],
                        embeddings=embeddings[j:j + BATCH],
                        documents=texts[j:j + BATCH],
                        metadatas=metadatas[j:j + BATCH],
                    )

                total_chunks += len(chunks)

            except Exception as e:
                if progress_callback:
                    progress_callback(i, len(files), f"SKIP {file_path.name}: {e}")

        return total_chunks

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Return the top-n most relevant code chunks for a query."""
        count = self.collection.count()
        if count == 0:
            return []

        n = min(n_results, count)
        embedding = self._embed([query])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
        )

        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text": doc,
                "file": meta["file"],
                "start_line": meta["start_line"],
            })
        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a readable context block."""
        if not chunks:
            return ""
        parts = []
        for chunk in chunks:
            header = f"# {chunk['file']} (line {chunk['start_line']})"
            parts.append(f"{header}\n```\n{chunk['text']}\n```")
        return "\n\n".join(parts)

    def clear(self) -> None:
        """Delete all indexed data."""
        self.client.delete_collection("codebase")
        self.collection = self.client.get_or_create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()
