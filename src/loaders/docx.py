from pathlib import Path
from langchain_core.documents import Document
from core.loader import BaseLoader
from docx import Document as DocxDocument
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table as DocxTable
from typing import Iterator
from docx.oxml.ns import qn

class DocxLoader(BaseLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path=file_path, extensions=[".docx"])
        self.style_to_md = {
            "Heading 1": "# ",
            "Heading 2": "## ",
            "Heading 3": "### ",
            "Heading 4": "#### ",
            "Title": "# ",
        }

    def _get_table_header(self, table: Table) -> list[str]:
        if len(table.rows) < 1:
            return []

        first_row = [cell.text.strip() for cell in table.rows[0].cells]
        first_row = list(dict.fromkeys(first_row))
        first_row = [c for c in first_row if c]

        return first_row

    def _is_data_table(self, header: list[str]) -> bool:
        return len(header) >= 2

    def _process_table(self, table: Table) -> str:
        headers = self._get_table_header(table)
        lines = []

        if self._is_data_table(headers):
            for row in table.rows[1:]:
                cells = list(dict.fromkeys([cell.text.strip() for cell in row.cells]))
                parts = []

                for h, c in zip(headers, cells):
                    if c:
                        parts.append(f"{h}: {c}")

                if parts:
                    lines.append(" | ".join(parts))
        else:
            for row in table.rows:
                cells = list(dict.fromkeys([cell.text.strip() for cell in row.cells]))
                cells = [c for c in cells if c]

                if cells:
                    lines.append(": ".join(cells))

        return "\n".join(lines)

    def _paragraph_to_md(self, paragraph: Paragraph) -> str:
        style = paragraph.style.name if paragraph.style else "Normal"
        text = ""

        if paragraph._p.pPr is not None and paragraph._p.pPr.numPr is not None:
            if "Number" in style:
                prefix = "1. "
            else:
                prefix = "- "
        else:
            prefix = self.style_to_md.get(style, "")
        
        for element in paragraph._p:
            if element.tag == qn("w:r"):
                run_text = element.findtext(qn("w:t"), "") or ""
                bold = element.find(f"{qn('w:rPr')}/{qn('w:b')}") is not None
                
                if bold and run_text.strip():
                    run_text = f"**{run_text}**"
                    
                text += run_text
            elif element.tag == qn("w:hyperlink"):
                for run in element.findall(qn("w:r")):
                    run_text = run.findtext(qn("w:t"), "") or ""
                    text += run_text
            elif element.tag == qn("w:ins"):
                for run in element.findall(qn("w:r")):
                    run_text = run.findtext(qn("w:t"), "") or ""
                    text += run_text
                    
        text = text.strip()
        
        if not text:
            return ""
        
        return f"{prefix}{text}"

    def lazy_load(self) -> Iterator[Document]:
        document = DocxDocument(self.file_path)
        properties = document.core_properties
        content = []
        metadata = {
            "source": Path(self.file_path).name,
            "file_size": Path(self.file_path).stat().st_size,
        }

        for element in document.element.body:
            if isinstance(element, CT_P):
                paragraph = Paragraph(element, document)
                md_text = self._paragraph_to_md(paragraph)

                if md_text:
                    content.append(md_text)

            elif isinstance(element, CT_Tbl):
                table = DocxTable(element, document)
                table_text = self._process_table(table)

                if table_text:
                    content.append(table_text)

        content = "\n".join(content)
        metadata["word_count"] = len(content.split())

        for attribute in ["title", "identifier", "author", "last_modified_by", "description", "subject", "keywords", "category", "version", "revision", "language"]:
            value = getattr(properties, attribute, None)

            if value:
                metadata[attribute] = str(value)

        for attribute in ["created", "modified", "last_printed"]:
            value = getattr(properties, attribute, None)

            if value:
                metadata[attribute] = value.isoformat()

        yield Document(page_content=content, metadata=metadata)
