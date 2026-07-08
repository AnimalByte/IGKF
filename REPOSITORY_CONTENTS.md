# Repository Contents

This private repository snapshot is intended to support audit and reproduction of the IGKF paper experiment.

Included:

- Source code for graph construction, GraphRAG querying, and evaluation
- `data/testset.jsonl`, the final annotated 205-question benchmark
- `evaluation/outputs/`, including raw generations, retrieval packages, OpenAI Batch inputs/outputs/metadata, normalized judge scores, statistics, tables, figures, and publication manifests
- Final blinded pairwise GraphRAG judge artifacts under `evaluation/outputs/pairwise_judge/`
- Paper-facing final pairwise tables under `evaluation/outputs/tables/`
- Paper-facing final pairwise figures 5 and 6 under `evaluation/outputs/figures/`
- `README.md`
- `REPRODUCING_EXPERIMENT.md`
- `requirements.txt`
- `environment.yml`
- setup and pipeline scripts

Excluded:

- Qwen GGUF model weights under `models/`
- Raw GWAS VCFs other than the benchmark JSONL
- Neo4j database files
- ChromaDB binary index files
- Downloaded ontologies/database resources
- PubMed abstract cache
- Local secrets and environment files

The repository is private because the audit trail contains raw model outputs, Batch IDs, and full evaluation artifacts.

Latest pairwise result included:

- Qwen + GraphRAG wins: 23/205
- GPT-5.4 mini + GraphRAG wins: 165/205
- Ties: 8/205
- Position-unstable cases: 9/205
- Parse/incomplete cases after targeted retries: 0
