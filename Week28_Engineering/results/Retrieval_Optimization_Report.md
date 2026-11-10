# Retrieval Optimization Report

## Client and problem

This is a simulated support client. Its search has to find the right manual passage even when two devices share an error code, or when a customer describes a problem in their own words.

## Corpus and method

I used 4 versioned Markdown documents, including one untrusted forum post. The benchmark has 8 development questions and 24 held-out questions. The evidence labels identify exact source spans so they still work when chunk IDs change. The dense backend was sentence-transformers/all-MiniLM-L6-v2 (quantized ONNX, 384 dimensions). The answer formatter stayed fixed.

R0 starts with dense retrieval and fixed 32-word chunks. R1 uses the chunk setup selected on development questions. R2 adds BM25 and RRF, R3 rescores the same candidate set with the controlled reranker, and R4 adds guarded query expansion. Each run keeps 12 candidates and evaluates the top 5.

## Chunk comparison on development questions

| Run | Evidence Recall@5 | MRR@5 | All evidence | P50 ms | P95 ms | Avg. context tokens |
|---|---:|---:|---:|---:|---:|---:|
| fixed32 | 0.875 | 0.625 | 0.875 | 4.837 | 5.526 | 151.4 |
| overlap32 | 0.875 | 0.562 | 0.875 | 4.490 | 5.019 | 156.2 |
| fixed64 | 1.000 | 0.792 | 1.000 | 4.712 | 5.330 | 285.6 |
| sections | 1.000 | 1.000 | 1.000 | 5.218 | 6.355 | 95.6 |

Chunk counts and indexed words for each setting are in `chunk_inventory.csv`. Overlapping character coverage in the selected context is recorded as the duplicate fraction.

## Held-out R0–R4 comparison

| Run | Evidence Recall@5 | MRR@5 | All evidence | P50 ms | P95 ms | Avg. context tokens |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.864 | 0.636 | 0.864 | 4.908 | 5.876 | 156.2 |
| R1 | 1.000 | 0.871 | 1.000 | 3.849 | 5.107 | 96.2 |
| R2 | 1.000 | 0.955 | 1.000 | 4.778 | 6.621 | 100.9 |
| R3 | 1.000 | 1.000 | 1.000 | 5.336 | 6.567 | 102.6 |
| R4 | 1.000 | 1.000 | 1.000 | 6.175 | 10.320 | 102.6 |

Recall measures how many labelled evidence spans are covered by the selected chunks, including spans covered across adjacent chunks. MRR counts the first result that contains a complete labelled span. Chunk-ID Recall is reported separately. Unanswerable queries are excluded from recall and MRR.

## Sparse versus dense on the selected chunk setup

| Run | Evidence Recall@5 | MRR@5 | All evidence | P50 ms | P95 ms | Avg. context tokens |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 0.909 | 0.856 | 0.909 | 0.445 | 0.590 | 83.9 |
| dense | 1.000 | 0.871 | 1.000 | 4.620 | 6.193 | 96.2 |
| hybrid | 1.000 | 0.955 | 1.000 | 3.801 | 6.349 | 100.9 |

## Expansion control

| Run | Evidence Recall@5 | MRR@5 | All evidence | P50 ms | P95 ms | Avg. context tokens |
|---|---:|---:|---:|---:|---:|---:|
| off | 1.000 | 1.000 | 1.000 | 5.336 | 6.567 | 102.6 |
| guarded | 1.000 | 1.000 | 1.000 | 6.175 | 10.320 | 102.6 |
| broad | 1.000 | 1.000 | 1.000 | 10.155 | 11.188 | 103.9 |

Guarded expansion leaves precise wording alone. The broad control adds generic wording and may drop syntax such as API parentheses. The traces record rewritten identifiers lost and every question where expansion reduced evidence recall. The original query remains a separate branch.

## Security check

- sparse_unfiltered: untrusted top-5 ranks [1]; untrusted excerpt cited: False.
- sparse_filtered: untrusted top-5 ranks []; untrusted excerpt cited: False.
- dense_unfiltered: untrusted top-5 ranks [1]; untrusted excerpt cited: False.
- dense_filtered: untrusted top-5 ranks []; untrusted excerpt cited: False.
- hybrid_unfiltered: untrusted top-5 ranks [1]; untrusted excerpt cited: False.
- hybrid_filtered: untrusted top-5 ranks []; untrusted excerpt cited: False.

Authority comes from the source manifest. Search score does not set trust. Identical chunks are deduplicated within each ranking. The answer stage excludes untrusted passages even when the audit deliberately disables the retrieval filter.

## Failure examples

- R0 Q03 (exact, partial_span_or_boundary): What does send_logs() do on AX-17? First result: AX-17 diagnostics.
- R0 Q09 (paraphrase, partial_span_or_boundary): How can I get the AX-17 diagnostic information out of the device? First result: AX-17 firmware 2.1 recovery.
- R0 Q20 (exact, partial_span_or_boundary): How much free space does AX-17 need before installing an update? First result: AX-17 battery and charging.

The full failure counts, slice scores and source spans are in `benchmark.json` and `retrieval_traces.jsonl`.

## Recommendation

I would start a pilot with **R1** and **sections**. This stage was selected on development evidence recall, complete coverage and MRR, with duplication and work as tie-breakers. Held-out evidence Recall@5 was 1.000 versus 0.864 for R0. MRR@5 was 0.871. It abstained on 2/2 unanswerable questions.

R3 reached MRR@5 of 1.000 on the held-out set, so I would also compare it with R1 in a fresh pilot where the first result matters. That observation did not change the development-based default.

These runs measure a small synthetic English corpus on one machine with an already-built index. The timing is retrieval only. Counts of embedding calls, lexical calls, scored pairs and indexed/context words are cost proxies. No API usage was billed. Evidence coverage does not measure whether a person found the quoted answer helpful. Before production I would test with independent real support queries, more unknown products and versions, and a wider set of malicious documents.
