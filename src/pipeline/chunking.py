import re
from config import ChunkingSettings
from langchain_core.documents import Document
import uuid

class Chunking:
    def __init__(self, settings: ChunkingSettings = ChunkingSettings()) -> None:
        self.settings = settings

    def _markdown_chunker(self, text: str, metadata: dict) -> list[Document]:
        chunks = []
        current = []
        current_size = 0
        h1, h2 = "", ""

        def flush():
            nonlocal current, current_size

            content = "\n".join(current).strip()

            if len(content) < 50:
                current = []
                current_size = 0
                return

            chunks.append(Document(
                page_content=content,
                metadata={**metadata, "h1": h1, "h2": h2, "chunk_id": str(uuid.uuid4())}
            ))

            current = []
            current_size = 0

        in_table = False

        for line in text.split("\n"):
            stripped = line.strip()

            if stripped.startswith("|"):
                in_table = True
            elif in_table and not stripped.startswith("|"):
                in_table = False

            if re.match(r"^# [^#]", stripped):
                if current_size > 100:
                    flush()

                h1 = stripped.lstrip("# ").strip()
                h2 = ""
                current.append(line)
                current_size = len(line)

            elif re.match(r"^## [^#]", stripped):
                if current_size > self.settings.chunk_size // 3:
                    flush()

                h2 = stripped.lstrip("# ").strip()
                current.append(line)
                current_size += len(line)

            elif re.match(r"^#{3,}", stripped):
                if current_size > self.settings.chunk_size // 2:
                    flush()

                current.append(line)
                current_size += len(line)

            else:
                current.append(line)
                current_size += len(line)

                if current_size > self.settings.chunk_size and not in_table:
                    flush()

        flush()
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        result = []

        for document in documents:
            result.extend(self._markdown_chunker(document.page_content, document.metadata))

        return result