"""Baseline ingestion: one LangChain Document per PDF page, no cleanup.

Deliberately naive. Boilerplate stripping, OCR fallback etc. get added
only when the eval shows they're needed (log each in docs/DECISIONS.md).
"""
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader


def load_corpus(data_dir):
    docs = []
    for p in sorted(Path(data_dir).glob("**/*.pdf")):
        for d in PyMuPDFLoader(str(p)).load():
            d.page_content = d.page_content.strip()
            # loader pages are 0-indexed; store 1-indexed so citations match the PDF viewer
            d.metadata = {"doc": p.name, "page": int(d.metadata.get("page", 0)) + 1}
            docs.append(d)
    return docs
