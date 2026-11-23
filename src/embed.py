import hashlib
import math
import re


class STEmbedder:
    def __init__(self, model_name, query_prefix=""):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def embed_docs(self, texts, batch_size=32):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

    def embed_query(self, query):
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0].tolist()


class FakeEmbedder:
    """Hashed bag-of-words. Only for testing the plumbing without model downloads."""
    DIM = 256

    def _vec(self, text):
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_docs(self, texts, batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)


def get_embedder(cfg):
    e = cfg["embedding"]
    if e["model"] == "fake":
        return FakeEmbedder()
    return STEmbedder(e["model"], e.get("query_prefix", ""))


# --- LangChain adapter -------------------------------------------------------
from langchain_core.embeddings import Embeddings  # noqa: E402


class LCEmbeddings(Embeddings):
    """Wraps our embedder so LangChain vector stores can use it."""

    def __init__(self, inner):
        self.inner = inner

    def embed_documents(self, texts):
        return self.inner.embed_docs(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


def get_lc_embeddings(cfg):
    return LCEmbeddings(get_embedder(cfg))
