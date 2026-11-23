"""python -m src.build_index  -> load, chunk, embed, store in Chroma (via LangChain)."""
from collections import defaultdict
from .config import load_config, ROOT
from .ingest import load_corpus
from .chunk import chunk_docs
from .vectorstore import get_vectorstore


def build_index(cfg):
    pages = load_corpus(ROOT / cfg["data_dir"])
    if not pages:
        raise SystemExit(f"No PDFs found in {cfg['data_dir']}/ -- add some first.")
    c = cfg["chunking"]
    chunks = chunk_docs(pages, c["strategy"], c["chunk_chars"], c["overlap_chars"], c["min_chars"])
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    # deterministic ids: doc::p<page>::c<n>  (handy for eval and debugging)
    counters, ids = defaultdict(int), []
    for ch in chunks:
        key = (ch.metadata["doc"], ch.metadata["page"])
        ids.append(f"{key[0]}::p{key[1]}::c{counters[key]}")
        counters[key] += 1

    get_vectorstore(cfg).delete_collection()   # rebuild from scratch
    vs = get_vectorstore(cfg)
    B = 500
    for i in range(0, len(chunks), B):
        vs.add_documents(chunks[i:i + B], ids=ids[i:i + B])
    print(f"Indexed {len(chunks)} chunks into '{cfg['collection']}'")


if __name__ == "__main__":
    build_index(load_config())
