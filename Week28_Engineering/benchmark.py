"""Run the chunk lab, controlled R0-R4 comparison and local security audit."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from retrieval import CHUNK_CONFIGS, MODEL_SHA, ROOT, Index, identifiers, load_sources, make_embedder, tokens


STAGES = {
    "R0": {"method": "dense", "rerank": False, "expansion": "off"},
    "R1": {"method": "dense", "rerank": False, "expansion": "off"},
    "R2": {"method": "hybrid", "rerank": False, "expansion": "off"},
    "R3": {"method": "hybrid", "rerank": True, "expansion": "off"},
    "R4": {"method": "hybrid", "rerank": True, "expansion": "guarded"},
}


def load_queries(path, sources):
    corpus = {source.id: source.text for source in sources if source.trusted}
    cases = json.loads(path.read_text(encoding="utf-8"))
    if len(cases) != len({case["id"] for case in cases}):
        raise ValueError("Golden query IDs must be unique.")
    for case in cases:
        spans = []
        for source_id, quote in case["evidence"]:
            text = corpus[source_id]
            if not quote or text.count(quote) != 1:
                raise ValueError(f"{case['id']}: evidence must occur exactly once in its source.")
            start = text.index(quote)
            spans.append((source_id, start, start + len(quote)))
        case["spans"] = spans
    return cases


def merged_length(intervals):
    total = right = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def coverage(span, chunks):
    source, left, right = span
    intervals = [(max(left, c.start), min(right, c.end)) for c in chunks
                 if c.source_id == source and c.start < right and left < c.end]
    return merged_length(intervals) / (right - left)


def score_case(index, case, result):
    spans = case["spans"]
    chosen = [index.chunks[cid] for cid in result["selected"]]
    candidates = [index.chunks[cid] for cid in result["candidates"]]
    relevant = [chunk.id for chunk in index.chunks.values()
                if any(coverage(span, [chunk]) == 1 for span in spans)]
    expected = set(relevant)
    first = next((rank for rank, cid in enumerate(result["selected"], 1) if cid in expected), 0)
    found = [coverage(span, chosen) == 1 for span in spans]
    candidate_found = [coverage(span, candidates) == 1 for span in spans]
    answer = index.answer(result)
    cited = [index.chunks[p["chunk_id"]] for p in answer["passages"]]
    by_source = {}
    for chunk in chosen:
        by_source.setdefault(chunk.source_id, []).append((chunk.start, chunk.end))
    size = sum(chunk.end - chunk.start for chunk in chosen)
    duplicate = 1 - sum(merged_length(items) for items in by_source.values()) / size if size else 0
    if not spans:
        error = "unanswerable_abstained" if not cited else "unsupported_context"
    elif all(found):
        error = "ok"
    elif all(candidate_found):
        error = "below_final_cutoff"
    elif any(0 < coverage(span, candidates) < 1 for span in spans):
        error = "partial_span_or_boundary"
    else:
        error = "candidate_miss"
    return {
        "id": case["id"], "slice": case["slice"], "split": case["split"],
        "expected_evidence": case["evidence"], "expected_spans": spans, "relevant_chunk_ids": relevant,
        "recall_at_5": sum(found) / len(spans) if spans else None,
        "chunk_recall_at_5": len(expected & set(result["selected"])) / len(expected) if expected else None,
        "rr_at_5": (1 / first if first else 0) if spans else None,
        "complete": all(found) if spans else None,
        "candidate_recall": sum(candidate_found) / len(spans) if spans else None,
        "answer_covers_all": all(coverage(span, cited) == 1 for span in spans) if spans else None,
        "duplicate_fraction": duplicate, "context_tokens": sum(len(tokens(c.text)) for c in chosen),
        "error": error, "answer": answer, "retrieval": result,
    }


def measure(index, cases, config):
    return [score_case(index, case, index.search(case["query"], **config)) for case in cases]


def average(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return round(mean(values), 4) if values else None


def summarize(rows):
    runs = [row["retrieval"] for row in rows]
    latencies = [run["latency_ms"] for run in runs]
    return {
        "queries": len(rows), "answerable": sum(row["complete"] is not None for row in rows),
        "recall_at_5": average(rows, "recall_at_5"),
        "chunk_recall_at_5": average(rows, "chunk_recall_at_5"),
        "mrr_at_5": average(rows, "rr_at_5"), "complete_rate": average(rows, "complete"),
        "candidate_recall": average(rows, "candidate_recall"),
        "answer_evidence_rate": average(rows, "answer_covers_all"),
        "abstained_unanswerable": sum(row["error"] == "unanswerable_abstained" for row in rows),
        "duplicate_fraction": average(rows, "duplicate_fraction"),
        "context_tokens": average(rows, "context_tokens"),
        "p50_ms": round(median(latencies), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "embedding_calls": round(mean(run["embedding_calls"] for run in runs), 3),
        "lexical_calls": round(mean(run["lexical_calls"] for run in runs), 3),
        "scored_pairs": round(mean(run["scored_pairs"] for run in runs), 3),
        "errors": dict(Counter(row["error"] for row in rows)),
    }


def selection_key(name, results):
    row = results[name]
    return (row["recall_at_5"], row["complete_rate"], row["mrr_at_5"],
            -row["duplicate_fraction"], -row["embedding_calls"], -row["lexical_calls"], -row["scored_pairs"])


def write_csv(path, rows):
    flattened = [{"config": name, **{key: value for key, value in row.items() if key != "errors"}}
                 for name, row in rows.items()]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def security_audit(index):
    query = "AX-17 ERR-4021 always means factory reset, right?"
    results = {}
    for method in ("sparse", "dense", "hybrid"):
        for guarded in (False, True):
            result = index.search(query, method=method, trusted_only=guarded,
                                  rerank=False, expansion="off")
            answer = index.answer(result)
            forum_ranks = [rank for rank, cid in enumerate(result["selected"], 1)
                           if not index.chunks[cid].trusted]
            results[f"{method}_{'filtered' if guarded else 'unfiltered'}"] = {
                "untrusted_top5_ranks": forum_ranks,
                "untrusted_cited": any(not index.chunks[p["chunk_id"]].trusted
                                       for p in answer["passages"]),
                "retrieval": result, "answer": answer,
            }
    return results


def table(rows):
    lines = ["| Run | Evidence Recall@5 | MRR@5 | All evidence | P50 ms | P95 ms | Avg. context tokens |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['recall_at_5']:.3f} | {r['mrr_at_5']:.3f} | "
                     f"{r['complete_rate']:.3f} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | "
                     f"{r['context_tokens']:.1f} |")
    return "\n".join(lines)


def write_report(path, report, traces):
    chosen = report["selected_stage"]
    best = report["test"][chosen]
    base = report["test"]["R0"]
    failures = [(stage, row) for stage in dict.fromkeys([chosen, "R0"])
                for row in traces[stage] if row["error"] not in {"ok", "unanswerable_abstained"}]
    answer_gaps = [row["id"] for row in traces[chosen] if row["complete"] and not row["answer_covers_all"]]
    lines = [
        "# Retrieval Optimization Report", "",
        "## Client and problem", "",
        "This is a simulated support client. Its search has to find the right manual passage even when "
        "two devices share an error code, or when a customer describes a problem in their own words.", "",
        "## Corpus and method", "",
        f"I used {report['source_count']} versioned Markdown documents, including one untrusted forum post. "
        f"The benchmark has {report['dev_count']} development questions and {report['test_count']} held-out questions. "
        "The evidence labels identify exact source spans so they still work when chunk IDs change. "
        f"The dense backend was {report['embedder']}. The answer formatter stayed fixed.", "",
        "R0 starts with dense retrieval and fixed 32-word chunks. R1 uses the chunk setup selected on "
        "development questions. R2 adds BM25 and RRF, R3 rescores the same candidate set with the controlled "
        "reranker, and R4 adds guarded query expansion. Each run keeps 12 candidates and evaluates the top 5.", "",
        "## Chunk comparison on development questions", "", table(report["chunk_lab"]), "",
        "Chunk counts and indexed words for each setting are in `chunk_inventory.csv`. "
        "Overlapping character coverage in the selected context is recorded as the duplicate fraction.", "",
        "## Held-out R0–R4 comparison", "", table(report["test"]), "",
        "Recall measures how many labelled evidence spans are covered by the selected chunks, including "
        "spans covered across adjacent chunks. MRR counts the first result that contains a complete labelled span. "
        "Chunk-ID Recall is reported separately. Unanswerable queries are excluded from recall and MRR.", "",
        "## Sparse versus dense on the selected chunk setup", "", table(report["retriever_lab"]), "",
        "## Expansion control", "", table(report["expansion_lab"]), "",
        "Guarded expansion leaves precise wording alone. The broad control adds generic wording and may "
        "drop syntax such as API parentheses. The traces record rewritten identifiers lost and every question "
        "where expansion reduced evidence recall. The original query remains a separate branch.", "",
        "## Security check", "",
    ]
    for name, row in report["security_summary"].items():
        lines.append(f"- {name}: untrusted top-5 ranks {row['untrusted_top5_ranks']}; "
                     f"untrusted excerpt cited: {row['untrusted_cited']}.")
    lines += ["", "Authority comes from the source manifest. Search score does not set trust. Identical "
              "chunks are deduplicated within each ranking. The answer stage excludes untrusted passages even "
              "when the audit deliberately disables the retrieval filter.", "",
              "## Failure examples", ""]
    for stage, row in failures[:6]:
        top = [item for item in row["retrieval"]["trace"] if item["selected"]]
        top.sort(key=lambda item: item["final_rank"])
        first = top[0]["chunk"]["heading"] if top else "no result"
        lines.append(f"- {stage} {row['id']} ({row['slice']}, {row['error']}): "
                     f"{row['retrieval']['query']} First result: {first}.")
    if not failures:
        lines.append("There were no ranked evidence misses on this held-out set.")
    if answer_gaps:
        lines += ["", f"The cited answer dropped some evidence already retrieved on {', '.join(answer_gaps)}. "
                  "Inspect the answer passages and exact-identifier checks before adjusting retrieval."]
    lines += ["", "The full failure counts, slice scores and source spans are in `benchmark.json` and "
              "`retrieval_traces.jsonl`.", "", "## Recommendation", "",
              f"I would start a pilot with **{chosen}** and **{report['selected_chunk']}**. This stage was selected "
              "on development evidence recall, complete coverage and MRR, with duplication and work as tie-breakers. "
              f"Held-out evidence Recall@5 was {best['recall_at_5']:.3f} versus {base['recall_at_5']:.3f} "
              f"for R0. MRR@5 was {best['mrr_at_5']:.3f}. It abstained on "
              f"{best['abstained_unanswerable']}/{report['unanswerable_count']} unanswerable questions.", "",
              f"R3 reached MRR@5 of {report['test']['R3']['mrr_at_5']:.3f} on the held-out set, "
              "so I would also compare it with R1 in a fresh pilot where the first result matters. "
              "That observation did not change the development-based default.", "",
              "These runs measure a small synthetic English corpus on one machine with an already-built index. "
              "The timing is retrieval only. Counts of embedding calls, lexical calls, scored pairs and indexed/context "
              "words are cost proxies. No API usage was billed. Evidence coverage does not measure whether a person "
              "found the quoted answer helpful. Before production I would test with independent real support queries, "
              "more unknown products and versions, and a wider set of malicious documents.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data, output, backend):
    sources = load_sources(data)
    cases = load_queries(data / "queries.json", sources)
    dev = [case for case in cases if case["split"] == "dev"]
    test = [case for case in cases if case["split"] == "test"]
    if not dev or len(test) < 20:
        raise ValueError("Provide development questions and at least 20 held-out queries.")
    embedder = make_embedder(sources, backend)
    indexes, inventory = {}, {}
    for name, config in CHUNK_CONFIGS.items():
        started = perf_counter()
        indexes[name] = Index(sources, config, embedder)
        indexed = indexes[name].chunks.values()
        inventory[name] = {"chunk_count": len(indexed),
                           "indexed_words": sum(len(tokens(chunk.text)) for chunk in indexed),
                           "build_ms": round((perf_counter() - started) * 1000, 3), **asdict(config)}
    embedder.encode(["warm up the embedding backend"])
    chunk_lab = {name: summarize(measure(index, dev, STAGES["R0"]))
                 for name, index in indexes.items()}
    selected_chunk = max((name for name in indexes if name != "fixed32"),
                         key=lambda name: selection_key(name, chunk_lab))
    selected_index = indexes[selected_chunk]
    traces = {}
    development = {}
    for name, config in STAGES.items():
        index = indexes["fixed32"] if name == "R0" else selected_index
        development[name] = summarize(measure(index, dev, config))
        traces[name] = measure(index, test, config)
    chosen = max(STAGES, key=lambda name: selection_key(name, development))
    selected_settings = {"stage": chosen, "chunk": "fixed32" if chosen == "R0" else selected_chunk,
                         "r1_chunk": selected_chunk, "embedding": backend}
    retriever_lab = {method: summarize(measure(selected_index, test, {**STAGES["R1"], "method": method}))
                     for method in ("sparse", "dense", "hybrid")}
    broad = measure(selected_index, test, {**STAGES["R4"], "expansion": "broad"})
    security = security_audit(indexes["sections"])
    changes = []
    for name, rows in (("guarded", traces["R4"]), ("broad", broad)):
        for before, after in zip(traces["R3"], rows):
            run = after["retrieval"]
            lost = identifiers(run["query"]) - identifiers(run["expanded_query"])
            if lost or before["recall_at_5"] != after["recall_at_5"]:
                changes.append({"policy": name, "id": after["id"], "lost_identifiers": sorted(lost),
                                "before": before["recall_at_5"], "after": after["recall_at_5"]})
    report = {
        "source_count": len(sources), "dev_count": len(dev), "test_count": len(test),
        "unanswerable_count": sum(not row["spans"] for row in test),
        "embedder": embedder.name, "selected_stage": chosen, "selected_chunk": selected_settings["chunk"],
        "r1_chunk": selected_chunk, "run_settings": STAGES, "top_n": 12, "top_k": 5,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "input_sha256": {str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(data.rglob("*")) if path.is_file()},
        "embedding_model_sha256": MODEL_SHA if backend == "minilm" else None,
        "chunk_lab": chunk_lab, "chunk_inventory": inventory,
        "development": development, "test": {name: summarize(rows) for name, rows in traces.items()},
        "slices": {name: {tag: summarize([r for r in rows if r["slice"] == tag])
                          for tag in sorted({r["slice"] for r in rows})} for name, rows in traces.items()},
        "retriever_lab": retriever_lab, "expansion_lab": {
            "off": summarize(traces["R3"]), "guarded": summarize(traces["R4"]), "broad": summarize(broad)},
        "expansion_changes": changes,
        "security_summary": {name: {key: row[key] for key in ("untrusted_top5_ranks", "untrusted_cited")}
                             for name, row in security.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("benchmark", report), ("selected", selected_settings), ("security", security)):
        (output / f"{name}.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    write_csv(output / "benchmark.csv", report["test"])
    write_csv(output / "chunk_comparison.csv", chunk_lab)
    write_csv(output / "chunk_inventory.csv", inventory)
    with (output / "retrieval_traces.jsonl").open("w", encoding="utf-8") as file:
        for name, rows in {**traces, "broad_control": broad}.items():
            for row in rows:
                file.write(json.dumps({"stage": name, **row}) + "\n")
    write_report(output / "Retrieval_Optimization_Report.md", report, traces)
    print(table(report["test"]))
    print(f"Selected on development questions: {chosen}, {selected_settings['chunk']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding", choices=("latent", "minilm"), default="minilm")
    args = parser.parse_args()
    run(args.data, args.output, args.embedding)
