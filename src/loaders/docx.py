from langchain_core.document_loaders import BaseLoader

class DocxLoader(BaseLoader):
    def lazy_load(self, file_path: str):
        print(file_path)