import pymupdf4llm
import pymupdf
from pathlib import Path
from typing import Iterator
from langchain_core.documents import Document
from core.loader import BaseLoader
import re

class PDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path=file_path, extensions=[".pdf"])

    def _parse_pdf_date(self, date_str: str) -> str | None:
        if not date_str:
            return None

        match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)

        if match:
            y, mo, d, h, mi, s = match.groups()

            return f"{y}-{mo}-{d}T{h}:{mi}:{s}"

        return None

    def lazy_load(self) -> Iterator[Document]:
        md_text = pymupdf4llm.to_markdown(
            str(self.file_path),
            force_text=True,
            ocr_languages=None,
            use_glyphs=False,
        )
        md_text = md_text.replace("<br>", " ")

        document = pymupdf.open(str(self.file_path))
        meta = document.metadata

        metadata = {
            "source": Path(self.file_path).name,
            "file_size": Path(self.file_path).stat().st_size,
            "page_count": len(document),
            "word_count": len(md_text.split()),
        }

        if meta.get("title"):
            metadata["title"] = meta["title"]
        if meta.get("author"):
            metadata["author"] = meta["author"]
        if meta.get("subject"):
            metadata["subject"] = meta["subject"]
        if meta.get("keywords"):
            metadata["keywords"] = meta["keywords"]
        if meta.get("creator"):
            metadata["creator"] = meta["creator"]
        if meta.get("creationDate"):
            parsed = self._parse_pdf_date(meta["creationDate"])

            if parsed:
                metadata["created"] = parsed
        if meta.get("modDate"):
            parsed = self._parse_pdf_date(meta["modDate"])

            if parsed:
                metadata["modified"] = parsed

        document.close()

        yield Document(page_content=md_text, metadata=metadata)