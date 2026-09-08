import re
import uuid
from config import ChunkingSettings
from langchain_core.documents import Document

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

class Chunking:
    def __init__(self, settings: ChunkingSettings = ChunkingSettings()) -> None:
        self.settings = settings

    @staticmethod
    def _split_by_sentences(line: str, budget: int) -> list[str]:
        sentences = SENTENCE_SPLIT_RE.split(line)
        pieces: list[str] = []
        buf = ""

        for s in sentences:
            if buf and len(buf) + 1 + len(s) > budget:
                pieces.append(buf)
                buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf:
            pieces.append(buf)

        return pieces or [line]

    def _merge_short_chunks(self, chunks: list[Document]) -> list[Document]:
        if not chunks:
            return chunks

        min_target = getattr(self.settings, "min_target_chars", self.settings.chunk_overlap * 2)
        merge_cap = int(self.settings.chunk_size * 1.3)

        merged: list[Document] = []
        for doc in chunks:
            is_table = doc.page_content.lstrip().startswith("|")
            prev = merged[-1] if merged else None

            can_merge = (
                prev is not None
                and not is_table
                and not prev.page_content.lstrip().startswith("|")
                and (len(doc.page_content) < min_target or len(prev.page_content) < min_target)
                and prev.metadata.get("h1") == doc.metadata.get("h1")
                and prev.metadata.get("h2") == doc.metadata.get("h2")
                and len(prev.page_content) + len(doc.page_content) <= merge_cap
            )

            if can_merge:
                doc_lines = doc.page_content.split("\n")

                while (
                    doc_lines
                    and not HEADER_RE.match(doc_lines[0].strip())
                    and doc_lines[0].strip()
                    and doc_lines[0].strip() in prev.page_content
                ):
                    doc_lines.pop(0)

                    if doc_lines and doc_lines[0].strip() == "":
                        doc_lines.pop(0)

                doc_content_clean = "\n".join(doc_lines).strip()

                combined_content = prev.page_content + "\n\n" + doc_content_clean
                combined_meta = dict(prev.metadata)
                combined_meta["h3"] = prev.metadata.get("h3") or doc.metadata.get("h3", "")
                merged[-1] = Document(page_content=combined_content, metadata=combined_meta)
            else:
                merged.append(doc)

        for i, d in enumerate(merged):
            d.metadata["chunk_index"] = i

        return merged

    def _markdown_chunker(self, text: str, metadata: dict) -> list[Document]:
        lines = text.split("\n")

        chunks: list[Document] = []
        current: list[str] = []
        current_size = 0
        overlap_only = False
        h1, h2, h3 = "", "", ""

        table_buffer: list[str] = []
        in_table = False

        def emit(content_lines: list[str], header_ctx: tuple[str, str, str]) -> None:
            content = "\n".join(content_lines).strip()

            if len(content) < self.settings.min_chunk_chars:
                return

            h1_, h2_, h3_ = header_ctx
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "h1": h1_,
                        "h2": h2_,
                        "h3": h3_,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks),
                    },
                )
            )

        def flush() -> None:
            nonlocal current, current_size, overlap_only

            if current:
                if overlap_only:
                    pass
                else:
                    emit(current, (h1, h2, h3))

            current = []
            current_size = 0
            overlap_only = False

        def flush_table() -> None:
            nonlocal table_buffer

            if not table_buffer:
                return

            max_table_chars = int(self.settings.chunk_size * self.settings.table_size_multiplier)
            joined = "\n".join(table_buffer)

            if len(joined) <= max_table_chars or len(table_buffer) < 3:
                emit(table_buffer, (h1, h2, h3))
                table_buffer = []
                return

            header_row, sep_row = table_buffer[0], table_buffer[1]
            data_rows = table_buffer[2:]

            piece: list[str] = [header_row, sep_row]
            piece_size = len(header_row) + len(sep_row)

            for row in data_rows:
                if piece_size + len(row) > self.settings.chunk_size and len(piece) > 2:
                    emit(piece, (h1, h2, h3))

                    piece = [header_row, sep_row]
                    piece_size = len(header_row) + len(sep_row)

                piece.append(row)
                piece_size += len(row)

            if len(piece) > 2:
                emit(piece, (h1, h2, h3))

            table_buffer = []

        def apply_overlap() -> None:
            nonlocal current, current_size, overlap_only

            if not chunks or self.settings.chunk_overlap <= 0:
                return

            prev = chunks[-1].page_content

            if prev.lstrip().startswith("|"):
                return

            prose_lines = [l for l in prev.split("\n") if not HEADER_RE.match(l.strip())]
            prose = "\n".join(prose_lines).strip()

            if not prose:
                return

            tail = prose[-self.settings.chunk_overlap * 3:]
            sentences = SENTENCE_SPLIT_RE.split(tail)
            overlap_text = ""

            for sentence in reversed(sentences):
                if len(overlap_text) + len(sentence) > self.settings.chunk_overlap:
                    break

                overlap_text = (sentence + " " + overlap_text).strip()
            if overlap_text:
                current.append(overlap_text)
                current_size += len(overlap_text)
                overlap_only = True

        for line in lines:
            stripped = line.strip()

            was_in_table = in_table
            if TABLE_ROW_RE.match(stripped) or (in_table and TABLE_SEP_RE.match(stripped)):
                in_table = True
            elif in_table and stripped == "":
                in_table = False
            elif in_table and not TABLE_ROW_RE.match(stripped):
                in_table = False

            if was_in_table and not in_table:
                flush_table()

            header_match = HEADER_RE.match(stripped)

            if in_table:
                table_buffer.append(line)
                continue

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                flush()

                if level == 1:
                    h1, h2, h3 = title, "", ""
                elif level == 2:
                    h2, h3 = title, ""
                else:
                    h3 = title

                apply_overlap()
                current.append(line)
                current_size += len(line)
                overlap_only = False

                continue

            remaining_budget = self.settings.chunk_size - current_size
            if len(line) > remaining_budget and len(stripped) > 0:
                for piece in self._split_by_sentences(line, self.settings.chunk_size):
                    if current and current_size + len(piece) > self.settings.chunk_size:
                        flush()
                        apply_overlap()

                    current.append(piece)
                    current_size += len(piece)
                    overlap_only = False
                continue

            current.append(line)
            current_size += len(line)
            overlap_only = False

            if current_size > self.settings.chunk_size:
                flush()
                apply_overlap()

        if in_table:
            flush_table()

        flush()

        return self._merge_short_chunks(chunks)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        result = []

        for document in documents:
            result.extend(self._markdown_chunker(document.page_content, document.metadata))

        return result