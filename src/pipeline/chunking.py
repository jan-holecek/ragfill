from config import ChunkingSettings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import uuid

class Chunking:
    def __init__(self, settings: ChunkingSettings = ChunkingSettings()) -> None:
        self.settings = settings

    def _markdown_header_text_chunking(self, document: Document) -> list[Document]:
        text_splitter = MarkdownHeaderTextSplitter(
            strip_headers=False,
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
            ]
        )

        return text_splitter.split_text(document.page_content)

    def _recursive_character_text_chunking(self, chunks: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        return text_splitter.split_documents(chunks)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        result = []

        for document in documents:
            md_chunks = self._markdown_header_text_chunking(document)

            for chunk in md_chunks:
                merged_metadata = {**document.metadata, **chunk.metadata}

                if len(chunk.page_content) > self.settings.chunk_size:
                    recursive_chunks = self._recursive_character_text_chunking([chunk])

                    for recursive_chunk in recursive_chunks:
                        recursive_chunk.metadata = {
                            **merged_metadata,
                            **recursive_chunk.metadata,
                            "chunk_id": str(uuid.uuid4())
                        }

                    result.extend(recursive_chunks)
                else:
                    chunk.metadata = {
                        **merged_metadata,
                        "chunk_id": str(uuid.uuid4())
                    }

                    result.append(chunk)

        return result