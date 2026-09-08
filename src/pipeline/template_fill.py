from pathlib import Path
from config import SearchSettings
from db.elasticsearch import ElasticSearchDB
from models.response import RAGResponse
from pipeline.embedding import Embedding
from pipeline.generation import Generation
from pipeline.query_rewrite import QueryRewrite
from docx import Document
from docx.oxml.ns import qn
import zipfile
import xml.etree.ElementTree as ET
import re
import time
import json
from pipeline.search import Search
from prompts import build_template_fill_prompt, build_template_fill_all_prompt
from models.response import TemplateFillResponse, PlaceholderResponse
from typing import Generator

class TemplateFill:
    def __init__(self, db: ElasticSearchDB, embedding: Embedding, generation: Generation, rewriter: QueryRewrite | None = None) -> None:
        self.embedding = embedding
        self.search = Search(db, SearchSettings(bm25_K=1, knn_K=1, num_candidates=50))
        self.generation = generation
        self.rewriter = rewriter
        self.ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    def _extract_placeholders(self, file_path: str) -> dict[str, str]:
        path = Path(file_path).absolute()
        placeholders = {}

        with zipfile.ZipFile(path, "r") as zip_file:
            if "word/comments.xml" not in zip_file.namelist():
                return {}

            comments_xml = zip_file.read("word/comments.xml")
            root = ET.fromstring(comments_xml)
            comments = {}

            for comment in root.findall("w:comment", self.ns):
                comment_id = comment.get(f"{{{self.ns['w']}}}id")
                comment_text = "".join(text.text or "" for text in comment.findall(".//w:t", self.ns))
                comments[comment_id] = comment_text

        doc = Document(path)
        body = doc.element.body

        for elem in body.iter():
            if elem.tag == qn("w:commentRangeStart"):
                comment_id = elem.get(qn("w:id"))

                if comment_id in comments:
                    parent = elem.getparent()

                    if parent is not None:
                        text = "".join(t.text or "" for t in parent.findall(f".//{qn('w:t')}"))
                        matches = re.findall(r"\{\{[^}]+\}\}", text)

                        for match in matches:
                            placeholders[match] = comments[comment_id]

        return placeholders

    def _generate_placeholder_content(self, placeholder: str, prompt: str) -> RAGResponse | None:
        rewrite_result = None
        rewritten_prompt = prompt

        if self.rewriter:
            rewrite_result = self.rewriter.rewrite(prompt)
            rewritten_prompt = rewrite_result.rewritten_query

        query_vector = self.embedding.embed_query(rewritten_prompt)
        search_chunks, used_queries = self.search.rewrite_and_search(rewritten_prompt, self.embedding)

        if not search_chunks:
            return None

        response = self.generation.generate(
            prompt,
            search_chunks,
            query_vector,
            rewrite_result,
            prompt_builder=build_template_fill_prompt
        )
        response.used_queries = used_queries

        return response

    def _generate_all_placeholders(self, placeholders: dict[str, str]):
        all_chunks = {}
        used_queries_by_placeholder = {}

        for placeholder, prompt in placeholders.items():
            results, used_queries = self.search.rewrite_and_search(prompt, self.embedding)
            used_queries_by_placeholder[placeholder] = used_queries

            for chunk in results:
                all_chunks[chunk.id] = chunk

        if not all_chunks:
            return {p: "" for p in placeholders}, 0, None, used_queries_by_placeholder

        chunks = list(all_chunks.values())
        fields_description = "\n".join([f'"{p}": "{prompt}"' for p, prompt in placeholders.items()])
        llm_instruction = f"Fill the document template placeholders based on the provided context according to these requested fields:\n{fields_description}"

        response = self.generation.generate(
            query=llm_instruction,
            chunks=chunks,
            prompt_builder=build_template_fill_all_prompt
        )

        content = response.answer.strip()
        content = re.sub(r"```json\s*|\s*```", "", content).strip()
        try:
            values = json.loads(content)

            return {k: str(v) for k, v in values.items()}, response.elapsed, response, used_queries_by_placeholder
        except Exception:
            return {p: "" for p in placeholders}, response.elapsed, None, used_queries_by_placeholder

    def fill_stream(self, file_path: str) -> Generator[PlaceholderResponse, None, None]:
        placeholders = self._extract_placeholders(file_path)

        for placeholder, prompt in placeholders.items():
            generated = self._generate_placeholder_content(placeholder, prompt)

            if generated:
                yield PlaceholderResponse(
                    placeholder=placeholder,
                    prompt=prompt,
                    answer=generated.answer,
                    completion_tokens=generated.completion_tokens,
                    prompt_tokens=generated.prompt_tokens,
                    elapsed=generated.elapsed,
                    rewrite=generated.rewrite,
                    chunks=generated.chunks,
                    used_queries=generated.used_queries,
                )
            else:
                yield PlaceholderResponse(
                    placeholder=placeholder,
                    prompt=prompt,
                    answer="",
                    completion_tokens=0,
                    prompt_tokens=0,
                    elapsed=0,
                )

    def fill(self, file_path: str) -> TemplateFillResponse:
        placeholders = self._extract_placeholders(file_path)
        start = time.time()
        values, elapsed, response, used_queries_by_placeholder = self._generate_all_placeholders(placeholders)

        results = [
            PlaceholderResponse(
                placeholder=p,
                prompt=placeholders[p],
                answer=values.get(p, ""),
                completion_tokens=response.completion_tokens if response else 0,
                prompt_tokens=response.prompt_tokens if response else 0,
                elapsed=elapsed,
                chunks=response.chunks if response else [],
                rewrite=response.rewrite if response else None,
                used_queries=used_queries_by_placeholder.get(p),
            )

            for p in placeholders
        ]

        return TemplateFillResponse(
            results=results,
            total_elapsed=round(time.time() - start, 2),
            total_completion_tokens=response.completion_tokens if response else 0,
            total_prompt_tokens=response.prompt_tokens if response else 0,
        )