from abc import abstractmethod
from typing import Iterator
from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader
from langchain_core.documents import Document
from pathlib import Path

class BaseLoader(LangchainBaseLoader):
    def __init__(self, file_path: str, extensions: list[str]) -> None:
        self.extensions = extensions
        path = Path(file_path).absolute()

        if not path.exists():
            raise FileNotFoundError(path, "File not found.")

        self.file_path = path

    @staticmethod
    def get_loader(file_path: str) -> LangchainBaseLoader:
        from loaders.docx import DocxLoader

        extensions = Path(file_path).suffix.lower()
        loaders = {
            ".docx": DocxLoader,
        }

        if extensions not in loaders:
            raise ValueError(f"Unsupported file extension: {extensions}. Supported extensions are: {list(loaders.keys())}")

        return loaders[extensions](file_path)

    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        pass

    def supported_extensions(self) -> list[str]:
        return self.extensions

    def load(self) -> list[Document]:
        return list(self.lazy_load())