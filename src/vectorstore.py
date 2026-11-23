from langchain_chroma import Chroma
from .config import ROOT
from .embed import get_lc_embeddings


def get_vectorstore(cfg):
    return Chroma(
        collection_name=cfg["collection"],
        embedding_function=get_lc_embeddings(cfg),
        persist_directory=str(ROOT / cfg["index_dir"]),
        collection_metadata={"hnsw:space": "cosine"},
    )
