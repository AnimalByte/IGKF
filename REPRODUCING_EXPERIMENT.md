# Reproducing the IGKF Paper Experiment

This guide describes how to reproduce the final four-condition IGKF experiment from the current repository state. It is intentionally separate from the manuscript and does not contain paper prose.

## Scope

The canonical experiment compares exactly four generation conditions on the annotated 205-question benchmark in `data/testset.jsonl`:

- `qwen3_8b_baseline`
- `qwen3_8b_igkf`
- `gpt_5_4_mini_baseline`
- `gpt_5_4_mini_igkf`

Primary paired comparisons:

- `qwen3_8b_baseline` vs `qwen3_8b_igkf`
- `gpt_5_4_mini_baseline` vs `gpt_5_4_mini_igkf`

The local model is Qwen3-8B in verified non-thinking mode. The commercial generator is `gpt-5.4-mini-2026-03-17` with reasoning effort `none`. The judge is `gpt-5.5` with reasoning effort `high`.

## Environment

Recommended setup:

```bash
chmod +x setup.sh
./setup.sh
```

For an exact environment snapshot from the completed run:

```bash
conda env create -f environment.yml
conda activate gwas-env
```

The setup script is the preferred path for a fresh machine because it also installs the SciSpaCy model and validates critical packages.

## Required External Assets

The repository does not include large model/database artifacts by default. For a clean reproduction, prepare:

- Neo4j via `docker compose up -d neo4j`
- VEP cache in `~/vep_data`
- GWAS VCF at `data/ukb-d-2395_1.vcf.gz`
- Qwen3 GGUF at `models/Qwen3-8B-Q4_K_M.gguf`
- OpenAI API key only when submitting or downloading Batch jobs

## Build the Graph

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gwas-env
docker compose up -d neo4j
./run_pipeline.sh
```

The current experiment uses the existing IGKF GraphRAG implementation. It does not add PageRank, Personalized PageRank, Steiner trees, GNNs, graph transformers, or new graph algorithms.

## Validate the Benchmark

```bash
python -m evaluation.cli validate-benchmark --force
```

Expected result:

- 205 total records
- 205 unique question IDs
- no blank questions
- no duplicated exact or normalized question text
- benchmark validation report at `evaluation/outputs/metadata/benchmark_validation_report.json`

## Dry Run

```bash
python -m evaluation.cli dry-run
```

This validates configuration and Qwen3 non-thinking chat-template support without calling OpenAI APIs.

## Local Qwen3 Generation

```bash
python -m evaluation.cli generate-local --conditions qwen3_8b_baseline
python -m evaluation.cli generate-local --conditions qwen3_8b_igkf
```

Outputs are written incrementally under:

```text
evaluation/outputs/raw_generations/
evaluation/outputs/retrieval_packages/
```

Existing per-question outputs are skipped unless `--force` is supplied.

## GPT-5.4 Mini Batch Generation

Prepare the Batch file:

```bash
python -m evaluation.cli prepare-gpt-batch \
  --conditions gpt_5_4_mini_baseline,gpt_5_4_mini_igkf
```

Inspect the printed request count, token estimate, and output-token budget. Then submit:

```bash
export OPENAI_API_KEY="..."
python -m evaluation.cli batch submit evaluation/outputs/batch_metadata/<generation_manifest>.json
python -m evaluation.cli batch status evaluation/outputs/batch_metadata/<generation_manifest>.json
python -m evaluation.cli batch download evaluation/outputs/batch_metadata/<generation_manifest>.json
python -m evaluation.cli batch normalize-generations evaluation/outputs/batch_metadata/<generation_manifest>.json
```

Do not delete the source JSONL, manifest, raw downloaded JSONL, or normalized generation files.

## GPT-5.5 Batch Judging

Prepare and submit Pass A:

```bash
python -m evaluation.cli prepare-judge-batch --pass-name judge_pass_a_general_quality
python -m evaluation.cli batch submit evaluation/outputs/batch_metadata/<pass_a_manifest>.json
python -m evaluation.cli batch status evaluation/outputs/batch_metadata/<pass_a_manifest>.json
python -m evaluation.cli batch download evaluation/outputs/batch_metadata/<pass_a_manifest>.json
python -m evaluation.cli batch normalize-judges evaluation/outputs/batch_metadata/<pass_a_manifest>.json --force
```

Prepare and submit Pass B:

```bash
python -m evaluation.cli prepare-judge-batch --pass-name judge_pass_b_igkf_specific
python -m evaluation.cli batch submit evaluation/outputs/batch_metadata/<pass_b_manifest>.json
python -m evaluation.cli batch status evaluation/outputs/batch_metadata/<pass_b_manifest>.json
python -m evaluation.cli batch download evaluation/outputs/batch_metadata/<pass_b_manifest>.json
python -m evaluation.cli batch normalize-judges evaluation/outputs/batch_metadata/<pass_b_manifest>.json --force
```

### Judge Retry Policy

If a Batch request completes but the raw response is incomplete or unparseable:

1. Do not rerun successful questions.
2. Identify failed `custom_id` values from the normalization summary.
3. Prepare a targeted retry with only those question IDs or answer IDs.
4. Increase `openai.judge_max_output_tokens` only for the retry if needed.
5. Normalize the retry output with `--force` so successful retry scores replace failed earlier attempts.
6. Preserve every raw Batch output and normalization summary.

The completed experiment required targeted judge retries because some `gpt-5.5` high-reasoning responses exhausted smaller output budgets before emitting JSON.

## Analysis, Tables, Figures, Manifest

```bash
python -m evaluation.cli analyze --force-manifest
```

This runs:

- retrieval evaluation status
- statistical analysis
- table generation
- figure generation
- publication artifact hash manifest

Key outputs:

```text
evaluation/outputs/statistics/judge_scores_long.csv
evaluation/outputs/statistics/condition_summary.csv
evaluation/outputs/statistics/wilcoxon_primary_comparisons.csv
evaluation/outputs/statistics/wilcoxon_interaction_tests.csv
evaluation/outputs/tables/
evaluation/outputs/figures/
evaluation/outputs/metadata/publication_archive_manifest.json
```

The primary statistical tests are paired Wilcoxon signed-rank tests. Holm correction is applied across the primary within-model comparisons.

Figures use deterministic bootstrap 95% confidence intervals and are emitted as PNG, SVG, and PDF.

## Known Limitations to Disclose

- Retrieval relevance metrics are not reported unless independent adjudicated relevance judgments exist.
- The current entity resolver can match broad biomedical terms to graph entities; this should be treated as a retrieval limitation.
- Multi-entity questions may not always receive explicit multi-entity graph reasoning in the current IGKF architecture.
- OpenAI model aliases can resolve to dated snapshots in Batch metadata; report the returned snapshot when available.
- Judge retries are part of the audit trail and should be preserved rather than hidden.

## Cleanup

Routine cleanup is intentionally conservative:

```bash
./clean.sh --yes
```

This preserves `evaluation/outputs/`. Do not delete paper experiment outputs unless they have been archived.
