import json
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import Handler, LocalGenerator, SearchApp
from benchmark import STAGES, coverage, load_queries, score_case, security_audit
from retrieval import CHUNK_CONFIGS, ROOT, ChunkConfig, Index, Source, chunk_document, expand, load_sources, make_embedder, rrf


class ChunkTests(unittest.TestCase):
    def source(self, text):
        return Source("test", "test.md", "Test", "1", True, text)

    def test_empty_short_and_long_overlap(self):
        self.assertEqual(chunk_document(self.source(""), CHUNK_CONFIGS["sections"]), [])
        self.assertEqual(len(chunk_document(self.source("one"), CHUNK_CONFIGS["fixed32"])), 1)
        source = self.source("# Heading\n" + " ".join(f"word{i}" for i in range(100)))
        fixed = chunk_document(source, ChunkConfig("fixed", 15))
        overlap = chunk_document(source, ChunkConfig("fixed", 15, 5))
        self.assertGreater(len(overlap), len(fixed))
        self.assertLess(overlap[1].start, overlap[0].end)
        self.assertEqual(coverage(("test", 0, len(source.text)), fixed), 1.0)
        for chunk in overlap:
            self.assertEqual(source.text[chunk.start:chunk.end], chunk.text)
        self.assertEqual(overlap, chunk_document(source, ChunkConfig("fixed", 15, 5)))

    def test_section_keeps_heading_and_evidence(self):
        text = "# Product\n\n## Firmware 2.2\nHold Power for 5 seconds.\n\n## Other\nUnrelated."
        chunks = chunk_document(self.source(text), CHUNK_CONFIGS["sections"])
        self.assertIn("Hold Power for 5 seconds.", chunks[1].text)
        self.assertEqual(chunks[1].heading, "Firmware 2.2")
        self.assertEqual(chunks[1].start_line, 3)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / "data"
        cls.sources = load_sources(cls.data)
        cls.index = Index(cls.sources, CHUNK_CONFIGS["sections"], make_embedder(cls.sources, "minilm"))
        cls.cases = {case["id"]: case for case in load_queries(cls.data / "queries.json", cls.sources)}

    def test_rrf_example_and_duplicates(self):
        ranking, scores = rrf([["A", "B", "B", "C"], ["B", "D", "A"]])
        self.assertEqual(ranking, ["B", "A", "D", "C"])
        self.assertAlmostEqual(scores["B"], 1 / 62 + 1 / 61)

    def test_rerank_only_reorders_fixed_candidates(self):
        query = self.cases["Q02"]["query"]
        before = self.index.search(query, **STAGES["R2"])
        after = self.index.search(query, **STAGES["R3"])
        self.assertEqual(before["candidates"], after["candidates"])
        self.assertTrue(set(after["selected"]) <= set(after["candidates"]))
        self.assertEqual(after["scored_pairs"], len(before["candidates"]))
        self.assertTrue(score_case(self.index, self.cases["Q02"], after)["answer_covers_all"])

    def test_guarded_expansion_and_drift_control(self):
        self.assertEqual(expand("AX-17 send_logs()", "guarded")[0], "AX-17 send_logs()")
        self.assertIn("account recovery", expand("How do I get back into my account?", "guarded")[0])
        self.assertNotIn("()", expand("AX-17 send_logs()", "broad")[0])

    def test_unanswerable_and_adversarial_queries(self):
        for key in ("Q19", "Q22"):
            run = self.index.search(self.cases[key]["query"], **STAGES["R3"])
            self.assertEqual(self.index.answer(run)["status"], "insufficient_evidence")
        audit = security_audit(self.index)
        self.assertTrue(audit["sparse_unfiltered"]["untrusted_top5_ranks"])
        self.assertFalse(audit["hybrid_filtered"]["untrusted_top5_ranks"])
        self.assertTrue(all(not row["untrusted_cited"] for row in audit.values()))

    def test_evidence_and_traces(self):
        case = self.cases["Q11"]
        result = self.index.search(case["query"], **STAGES["R3"])
        row = score_case(self.index, case, result)
        self.assertEqual(row["recall_at_5"], 1)
        self.assertGreaterEqual(len(row["relevant_chunk_ids"]), 2)
        self.assertTrue(any(hit["sparse_ranks"] and hit["dense_ranks"] for hit in result["trace"]))
        self.assertTrue(all("citation" in passage for passage in row["answer"]["passages"]))

    def test_server_normal_and_invalid_requests(self):
        selection = {"stage": "R1", "chunk": "sections", "r1_chunk": "sections", "embedding": "minilm"}
        app = SearchApp(self.data, selection, "minilm")
        with ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, app=app)) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base, timeout=5) as page:
                    self.assertIn(b"Week 28 Manual Search", page.read())
                with urlopen(base + "/api/config", timeout=5) as response:
                    self.assertEqual(json.load(response)["stage"], "R1")
                for query, expected in ((self.cases["Q02"]["query"], 200), ("", 400)):
                    request = Request(base + "/api/search", json.dumps({"query": query}).encode(),
                                      headers={"Content-Type": "application/json"})
                    if expected == 400:
                        with self.assertRaises(HTTPError) as raised:
                            urlopen(request, timeout=5)
                        self.assertEqual(raised.exception.code, 400)
                    else:
                        with urlopen(request, timeout=5) as response:
                            data = json.load(response)
                        self.assertEqual(data["stage"], "R1")
                        self.assertTrue(data["trace"])
                        self.assertIn("temperature sensor failed", data["answer"]["answer"])
            finally:
                server.shutdown()
                thread.join()

    def test_generator_endpoint_is_local(self):
        with self.assertRaises(ValueError):
            LocalGenerator("https://example.com/v1/chat/completions", "test")
        LocalGenerator("http://127.0.0.1:1234/v1/chat/completions", "test")


if __name__ == "__main__":
    unittest.main()
