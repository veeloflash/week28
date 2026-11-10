# Week 28 Engineering

This project searches product manuals and shows the exact passages it used. You can switch retrieval stages in the browser to see what changes in the ranking.

## Run it on Windows

The saved Week 28 folder already includes the embedding model and Python 3.12 runtime dependencies. Open PowerShell and run:

```powershell
Set-Location -LiteralPath 'D:\作品\第28周\Week28_Engineering'
.\run.ps1
```

Open [Manual Search](http://127.0.0.1:8028), then try **What should I do for ERR-4021 on AX-18?** The cited answer should point to the AX-18 service manual and explain that the temperature sensor failed. Use the retrieval trace to see why each passage appeared. Press Ctrl+C in PowerShell to stop the server.

For a fresh copy without the bundled dependencies or model:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Week28_Engineering\requirements.txt
.\Week28_Engineering\fetch_model.ps1
.\Week28_Engineering\run.ps1
```

The downloader checks both files against pinned hashes. [MiniLM model files](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/5641a7880f40ebf4035d05e60c5f9b7a9c272c84/onnx)

## What the stages change

The app reads the source manifest and splits the manuals into chunks with source IDs and line numbers. MiniLM ranks passages by meaning. Hybrid search adds BM25 keyword matching, and reciprocal rank fusion (RRF) combines the two ranked lists. Reranking uses a deterministic relevance score on the existing shortlist. Guarded expansion adds known aliases while preserving precise queries.

| Stage | Change |
|---|---|
| R0 | Dense search over fixed 32-word chunks |
| R1 | Dense search over Markdown sections |
| R2 | Add BM25 and RRF |
| R3 | Rerank the same 12 candidates before keeping five |
| R4 | Add guarded query expansion |

The answer quotes trusted passages and shows their sources. If you already run a local chat model, you can enable its optional summary with `--generator-url` and `--generator-model`. The URL must be a local `/v1/chat/completions` endpoint. The cited excerpts remain visible alongside any summary.

## What I found

I used eight development questions to choose the default and 24 separate questions to evaluate it. Among the 22 answerable evaluation questions, R1 raised evidence Recall@5 from 86.4% to 100%. R1 is the default because it tied the more complex stages on the development questions and required less work. R3 moved relevant evidence to the first result on all 22 answerable evaluation questions, reaching MRR@5 of 1.000. R4 did not improve that score on this set.

In the security test, the fake forum post ranked first in each unfiltered retrieval mode. The trusted-source filter removed it from the top results, and the answer formatter excluded it from citations even in the unfiltered audit. The corpus is small, synthetic and English-only, so these scores describe this experiment.

## Reproduce the results

From the saved folder, use the prepared runtime:

```powershell
$env:PYTHONPATH = (Resolve-Path .\.runtime).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
& $python -m unittest -v
& $python benchmark.py --output .\results-rerun
```

On a fresh copy, use the Python executable in `.venv\Scripts` and skip the `PYTHONPATH` line.

- [Optimization report](results/Retrieval_Optimization_Report.md) explains the experiments and recommendation.
- [Benchmark table](results/benchmark.csv) compares R0–R4.
- [Chunk comparison](results/chunk_comparison.csv), [query traces](results/retrieval_traces.jsonl), and [security audit](results/security.json) provide the supporting results.
