"""Small, inspectable retrieval pipeline for the Week 28 support manuals."""
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.I)
HEADING = re.compile(r"^#{1,6}[ \t]+(.+)$", re.M)
IDENTIFIER = re.compile(r"\b(?:AX|ERR)[ -]?\d+\b|\b[a-z]+_[a-z_]+\(\)", re.I)
STOP = set("a an and are can do for how i is it me my on or should the to was what when where which with".split())
ALIASES = {
    "screen": "display", "scren": "display", "dark": "blank", "blnk": "blank",
    "startup": "boot", "cabel": "cable", "unpluged": "disconnected",
    "postage": "shipping", "saturday": "weekend", "internet": "connection",
    "attachment": "upload", "attachments": "upload", "forgot": "forgotten",
}
EXPANSIONS = {
    "turns on": "boot", "turning it on": "boot", "screen stays dark": "blank display",
    "send back": "return", "no connection": "offline", "postage": "shipping label",
    "get back into my account": "account recovery", "ticket i sent on saturday": "weekend support",
}
MODEL_SHA = "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
TOKENIZER_SHA = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"


def tokens(text):
    return WORD.findall(text.casefold())


def identifiers(text):
    found = set()
    for value in IDENTIFIER.findall(text.casefold()):
        if value.startswith(("ax", "err")) and not value.endswith("()"):
            value = re.sub(r"^(ax|err)[ -]?", r"\1-", value)
        found.add(value)
    return found


@dataclass(frozen=True)
class Source:
    id: str
    path: str
    title: str
    version: str
    trusted: bool
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    path: str
    title: str
    version: str
    trusted: bool
    heading: str
    start: int
    end: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class ChunkConfig:
    method: str
    size: int
    overlap: int = 0


CHUNK_CONFIGS = {
    "fixed32": ChunkConfig("fixed", 32),
    "overlap32": ChunkConfig("fixed", 32, 8),
    "fixed64": ChunkConfig("fixed", 64),
    "sections": ChunkConfig("sections", 55),
}


def load_sources(data_dir):
    data_dir = Path(data_dir).resolve()
    manifest = json.loads((data_dir / "sources.json").read_text(encoding="utf-8"))
    sources = []
    for entry in manifest:
        path = (data_dir / entry["path"]).resolve()
        if not path.is_relative_to(data_dir) or path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("Source paths must be .md or .txt files inside the data directory.")
        if type(entry["trusted"]) is not bool:
            raise ValueError("The trusted flag must be true or false.")
        sources.append(Source(**entry, text=path.read_text(encoding="utf-8")))
    if not sources or len({source.id for source in sources}) != len(sources):
        raise ValueError("The manifest needs unique source IDs.")
    return sources


def section_ranges(text):
    headings = list(HEADING.finditer(text))
    if not headings:
        return [(0, len(text), "")]
    ranges = [(0, headings[0].start(), "")] if headings[0].start() else []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        ranges.append((heading.start(), end, heading[1].strip()))
    return ranges


def chunk_document(source, config):
    if config.method not in {"fixed", "sections"} or not 0 <= config.overlap < config.size:
        raise ValueError("Invalid chunk configuration.")
    sections = section_ranges(source.text)
    ranges = sections if config.method == "sections" else [(0, len(source.text), "")]
    chunks = []
    for left, right, heading in ranges:
        matches = list(WORD.finditer(source.text, left, right))
        for offset in range(0, len(matches), config.size - config.overlap):
            end_index = min(len(matches), offset + config.size)
            start = left if offset == 0 else matches[offset].start()
            end = right if end_index == len(matches) else matches[end_index].start()
            raw = source.text[start:end]
            label = heading or next((name for a, b, name in sections if a <= start < b), "")
            last_text = start + len(raw.rstrip())
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{source.id}:{source.version}:{start}-{end}:{digest}"
            chunks.append(Chunk(chunk_id, source.id, source.path, source.title, source.version,
                                source.trusted, label, start, end,
                                source.text.count("\n", 0, start) + 1,
                                source.text.count("\n", 0, last_text) + 1, raw))
            if end_index == len(matches):
                break
    return chunks


class BM25:
    def __init__(self, chunks):
        self.counts = [Counter(tokens(chunk.text)) for chunk in chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(chunks)
        frequency = Counter(term for count in self.counts for term in count)
        self.idf = {
            term: math.log1p((len(chunks) - count + .5) / (count + .5))
            for term, count in frequency.items()
        }

    def scores(self, query):
        terms = set(tokens(query)) - STOP
        scores = []
        for count, length in zip(self.counts, self.lengths):
            normalizer = 1.5 * (.25 + .75 * length / self.average)
            scores.append(sum(self.idf[term] * 2.5 * count[term] /
                              (count[term] + normalizer) for term in terms if count[term]))
        return scores


def unit_vectors(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class LatentEmbedder:
    """Local TF-IDF -> truncated SVD vectors. Fit once on trusted source sections."""

    def __init__(self, sources):
        sections = [source.text[a:b] for source in sources if source.trusted
                    for a, b, _ in section_ranges(source.text) if source.text[a:b].strip()]
        term_sets = [set(self.terms(text)) for text in sections]
        counts = Counter(term for terms in term_sets for term in terms)
        self.vocab = {term: i for i, term in enumerate(sorted(counts))}
        self.idf = np.array([math.log((1 + len(sections)) / (1 + counts[t])) + 1
                             for t in self.vocab])
        if not self.vocab:
            raise ValueError("Add a trusted source with indexable text.")
        _, _, axes = np.linalg.svd(self.matrix(sections), full_matrices=False)
        self.axes = axes[:min(16, len(axes))].T
        self.name = "local TF-IDF + SVD (16 dimensions maximum, domain aliases)"

    @staticmethod
    def terms(text):
        return [ALIASES.get(t, t) for t in tokens(text) if t not in STOP]

    def matrix(self, texts):
        rows = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            for term, count in Counter(self.terms(text)).items():
                if term in self.vocab:
                    rows[i, self.vocab[term]] = 1 + math.log(count)
        return unit_vectors(rows * self.idf)

    def encode(self, texts):
        return unit_vectors(self.matrix(texts) @ self.axes)


class MiniLMEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2 (quantized ONNX, 384 dimensions)"

    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model = model_dir / "model.onnx"
        if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError("The MiniLM model checksum does not match the pinned model.")
        tokenizer = model_dir / "tokenizer.json"
        if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA:
            raise ValueError("The MiniLM tokenizer checksum does not match the pinned tokenizer.")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        mask = np.asarray([e.attention_mask for e in encoded], dtype=np.int64)
        inputs = {"input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                  "attention_mask": mask,
                  "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64)}
        hidden = self.session.run(None, {k: v for k, v in inputs.items() if k in self.input_names})[0]
        weights = mask[:, :, None]
        return unit_vectors((hidden * weights).sum(axis=1) / weights.sum(axis=1))


def make_embedder(sources, backend):
    if backend == "minilm":
        return MiniLMEmbedder(ROOT / ".models")
    if backend == "latent":
        return LatentEmbedder(sources)
    raise ValueError(f"Unknown embedding backend: {backend}")


def rrf(rankings, constant=60):
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (constant + rank)
    return sorted(scores, key=lambda key: (-scores[key], key)), dict(scores)


def expand(query, mode):
    if mode == "off":
        return query, "off"
    precise = identifiers(query) or re.search(r"\d|\"|\b(?:exactly|only|without|never)\b", query, re.I)
    if precise and mode == "guarded":
        return query, "kept precise wording"
    if precise and mode == "broad":
        return "device help " + " ".join(tokens(query)), "broad control added generic terms"
    extra = [value for phrase, value in EXPANSIONS.items() if phrase in query.casefold()]
    return (query + " " + " ".join(extra), "added domain terms") if extra else (query, "no matching alias")


def rerank_score(query, chunk, dense, lexical):
    q_ids, c_ids = identifiers(query), identifiers(chunk.text)
    exact = len(q_ids & c_ids) / max(len(q_ids), 1)
    q_terms = set(tokens(query)) - STOP
    coverage = len(q_terms & set(tokens(chunk.text))) / max(len(q_terms), 1)
    heading = len(q_terms & set(tokens(chunk.heading))) / max(len(q_terms), 1)
    wrong_product = any(x.startswith("ax-") for x in q_ids) and any(
        x.startswith("ax-") for x in c_ids) and not {
        x for x in q_ids & c_ids if x.startswith("ax-")}
    return float(dense + .30 * coverage + .10 * heading + .28 * exact
                 + .04 * math.log1p(lexical) - .25 * bool(wrong_product))


class Index:
    def __init__(self, sources, config, embedder):
        self.embedder = embedder
        self.chunks = {chunk.id: chunk for source in sources for chunk in chunk_document(source, config)}
        if not self.chunks:
            raise ValueError("The selected corpus has no indexable text.")
        self.ids = list(self.chunks)
        chunks = list(self.chunks.values())
        self.bm25 = BM25(chunks)
        self.vectors = np.concatenate([embedder.encode([chunk.text for chunk in chunks[i:i + 32]])
                                       for i in range(0, len(chunks), 32)])

    def search(self, query, *, method="hybrid", rerank=True, expansion="guarded",
               trusted_only=True, top_n=12, top_k=5):
        if not query.strip() or len(query) > 1000 or not 0 < top_k <= top_n:
            raise ValueError("Enter a query of 1 to 1000 characters and use 0 < top-k <= top-N.")
        if method not in {"dense", "sparse", "hybrid"} or expansion not in {"off", "guarded", "broad"}:
            raise ValueError("Unknown retrieval method or expansion mode.")
        started = perf_counter()
        rewritten, reason = expand(query, expansion)
        forms = [query] if rewritten == query else [query, rewritten]
        eligible = [i for i, cid in enumerate(self.ids) if self.chunks[cid].trusted or not trusted_only]
        dense_lists, sparse_lists = [], []
        dense_scores, sparse_scores = {}, {}
        def rank(scores, positive=False):
            order = sorted((i for i in eligible if not positive or scores[i] > 0),
                           key=lambda i: (-scores[i], self.ids[i]))
            unique = {}
            for i in order:
                key = tuple(tokens(self.chunks[self.ids[i]].text))
                unique.setdefault(key, self.ids[i])
            return list(unique.values())[:top_n]

        for form in forms:
            if method != "sparse":
                values = self.vectors @ self.embedder.encode([form])[0]
                dense_lists.append(rank(values))
                for cid, value in zip(self.ids, values):
                    dense_scores[cid] = max(float(value), dense_scores.get(cid, -1.0))
            if method != "dense":
                values = self.bm25.scores(form)
                sparse_lists.append(rank(values, positive=True))
                for cid, value in zip(self.ids, values):
                    sparse_scores[cid] = max(value, sparse_scores.get(cid, 0.0))
        ranked, fused_scores = rrf(dense_lists + sparse_lists)
        candidates = ranked[:top_n]
        reranked = {}
        if rerank:
            reranked = {cid: rerank_score(query, self.chunks[cid], dense_scores.get(cid, 0),
                                          sparse_scores.get(cid, 0)) for cid in candidates}
            final = sorted(candidates, key=lambda cid: (-reranked[cid], candidates.index(cid)))
        else:
            final = candidates
        dense_ranks = [{cid: rank for rank, cid in enumerate(items, 1)} for items in dense_lists]
        sparse_ranks = [{cid: rank for rank, cid in enumerate(items, 1)} for items in sparse_lists]
        final_ranks = {cid: rank for rank, cid in enumerate(final, 1)}
        trace = [{
            "chunk": asdict(self.chunks[cid]), "dense_score": dense_scores.get(cid),
            "sparse_score": sparse_scores.get(cid),
            "dense_ranks": [ranks.get(cid) for ranks in dense_ranks],
            "sparse_ranks": [ranks.get(cid) for ranks in sparse_ranks],
            "fused_rank": rank, "rrf_score": fused_scores[cid],
            "rerank_score": reranked.get(cid), "final_rank": final_ranks.get(cid),
            "selected": cid in final[:top_k],
        } for rank, cid in enumerate(ranked, 1)]
        return {
            "query": query, "expanded_query": rewritten, "expansion_reason": reason,
            "query_forms": forms, "trusted_only": trusted_only, "method": method,
            "reranker": "controlled relevance scorer" if rerank else "off",
            "embedding_calls": len(dense_lists), "lexical_calls": len(sparse_lists),
            "scored_pairs": len(reranked), "top_n": top_n, "top_k": top_k,
            "candidates": candidates, "selected": final[:top_k], "trace": trace,
            "latency_ms": (perf_counter() - started) * 1000,
        }

    def answer(self, result):
        """Quote only eligible manual passages and keep citations attached."""
        required = identifiers(result["query"])
        passages = []
        covered = set()
        seen = set()
        for row in sorted((r for r in result["trace"] if r["selected"]), key=lambda r: r["final_rank"]):
            chunk = self.chunks[row["chunk"]["id"]]
            found = identifiers(chunk.text)
            query_products = {value for value in required if value.startswith("ax-")}
            query_codes = required - query_products
            if not chunk.trusted or (query_products and not query_products & found):
                continue
            if query_codes and not query_codes & found:
                continue
            if not required and (row["dense_score"] or 0) < .28 and (row["sparse_score"] or 0) < 1.0:
                continue
            content_key = tuple(tokens(chunk.text))
            if content_key in seen:
                continue
            seen.add(content_key)
            covered.update(found)
            passages.append({"text": chunk.text.strip(), "chunk_id": chunk.id, "source_id": chunk.source_id,
                             "citation": f"{chunk.title} v{chunk.version}, {chunk.path}:{chunk.start_line}-{chunk.end_line}"})
        if not passages or not required <= covered:
            return {"status": "insufficient_evidence", "passages": [],
                    "answer": "I could not find enough matching evidence in the trusted manuals."}
        return {"status": "evidence_found", "passages": passages,
                "answer": "\n\n".join(f"{p['text']}\n[{p['citation']}]" for p in passages)}
