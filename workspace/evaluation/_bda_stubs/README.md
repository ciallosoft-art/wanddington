# Import stubs for the controlled BioDiscoveryAgent run

BDA's `gene.py` / `get_lit_review.py` import optional tool dependencies at module load time
(`langchain`, `scholarly`, `Bio`, `openai`, `transformers`, `html2text`, `paperqa`, `paper_scrapper`),
so the process cannot start without them — even in the **no-tools** configuration we evaluate, where
none of them is ever called.

These stubs satisfy the imports and nothing else. Every attribute they expose raises on use, so if a
run ever did reach a real tool call it would fail loudly rather than silently returning empty results
and being scored as a legitimate agent decision.

They are on `PYTHONPATH` only for the subprocess launched by `run_bda_controlled.py`.
