# Integrative Genomic Knowledge Framework (IGKF)
*A Knowledge-Graph-Powered RAG System for GWAS Data*

---

## Abstract

Genome-Wide Association Studies (GWAS) are instrumental in identifying genetic variants associated with complex traits and diseases. However, a significant challenge lies in interpreting these findings to understand the underlying biological mechanisms. This project presents a complete, reproducible pipeline for constructing a multi-layered biomedical knowledge graph from raw GWAS summary statistics and leveraging it within a Retrieval-Augmented Generation (RAG) system.

By integrating:

- GWAS variants and loci
- Functional annotations and pathways
- Ontologies (GO, HPO, Reactome, ClinVar, etc.)
- Drug–gene interactions, PPI data, and Open Targets disease/drug evidence
- Entities and relations extracted from PubMed abstracts

the system allows researchers to ask complex, natural language questions and receive synthesized, evidence-based answers with explicit provenance.

A canonical paper evaluation pipeline under `evaluation/` compares Qwen3-8B and GPT-5.4 mini with and without IGKF context while preserving raw generations, judge outputs, retrieval metadata, statistics, tables, figures, and reproducibility metadata.

---

## Dataset

This configuration is set up for a concrete, fully public example:

- **Trait:** Hair/balding pattern: Pattern 1  
- **IEU OpenGWAS ID:** `ukb-d-2395_1`  
- **Source:** https://gwas.mrcieu.ac.uk/datasets/ukb-d-2395_1/

The pipeline starts from the corresponding GWAS summary statistics (VCF), filters significant variants, and propagates them through the knowledge graph construction steps.

---

## System Architecture

IGKF implements a modular GraphRAG stack:

1. **Knowledge Graph (Neo4j Community 5.18.0)**  
   Stores:
   - GWAS variants and sentinel SNPs  
   - Genes and regulatory annotations  
   - Phenotypes & ontological concepts  
   - Pathways, clinical significance, drugs, PPIs, etc.

2. **Vector Store (ChromaDB 1.3.4)**  
   - Embeddings of PubMed abstracts and related textual evidence  
   - Supports semantic retrieval of literature evidence for graph entities.

3. **RAG API + Query Engine**
   - `api.py`, `query_engine.py`
   - Serves endpoints and an interactive interface that:
     - Retrieves subgraphs and embeddings
     - Feeds them into a local LLM or API-based model
     - Returns grounded, cited answers.

4. **Paper Evaluation Pipeline**
   - `evaluation/run_generations.py`
   - `evaluation/judge_answers.py`
   - `evaluation/evaluate_retrieval.py`
   - `evaluation/statistical_analysis.py`
   - `evaluation/make_tables.py`
   - `evaluation/make_figures.py`
   - Compares exactly four conditions:
     - `qwen3_8b_baseline`
     - `qwen3_8b_igkf`
     - `gpt_5_4_mini_baseline`
     - `gpt_5_4_mini_igkf`

---

## Key Dependencies

This project uses a carefully versioned environment for reproducibility:

### Core Scientific Stack
- **Python**: 3.10
- **NumPy**: 1.26.4
- **Pandas**: 2.0.3
- **SciPy**: 1.11.4

### Machine Learning & NLP
- **PyTorch**: 2.6.0 (CUDA 12.4 build in the tested environment)
- **Transformers**: 4.57.1
- **Sentence Transformers**: 5.1.2
- **spaCy**: 3.7.4
- **SciSpaCy**: 0.5.5 (with en_core_sci_lg 0.5.4 model)

### Graph & Vector Databases
- **Neo4j**: 5.18.0 (Community Edition via Docker)
- **ChromaDB**: 1.3.4

### LangChain Ecosystem
- **LangChain**: 0.3.27
- **LangChain Core**: 0.3.80
- **LangChain Community**: 0.3.31
- **LangChain OpenAI**: 0.3.35

### Evaluation & Datasets
- **RAGAS**: 0.4.3
- **Datasets**: 4.4.1 (HuggingFace)
- **Matplotlib**: 3.10.9 (publication figures)

### Bioinformatics
- **Biopython**: 1.86
- **Pronto**: 2.7.2 (for ontology parsing)
- **CrossMap**: (from bioconda, for genome liftover)
- **bx-python**, **pyBigWig**, **pysam** (from bioconda)

---

## Project Structure

After setup, the repository is expected to look like:

```text
Integrative-Genomic-Knowledge-Framework-IGKF-main/
├── data/
│   ├── testset.jsonl                    # Final annotated 205-question benchmark
│   └── ukb-d-2395_1.vcf.gz              # Original GWAS VCF (downloaded)
├── models/
│   └── Qwen3-8B-Q4_K_M.gguf             # Local Qwen3-8B GGUF model
├── results/                             # Legacy/generated outputs
├── evaluation/
│   ├── configs/paper_experiment.yaml    # Canonical experiment configuration
│   └── outputs/                         # Raw generations, batches, scores, stats, tables, figures
├── downloads/                           # Intermediate downloaded resources (ontologies, etc.)
├── neo4j/
│   ├── data/                            # Neo4j database (Docker volume)
│   ├── logs/                            # Neo4j logs
│   ├── import/                          # Import directory (if needed)
│   └── plugins/                         # APOC / n10s plugins (from image)
├── backups/                             # Optional Neo4j backup volume
├── docker-compose.yml                   # Neo4j Community configuration
├── README.md                            # This file
├── REPRODUCING_EXPERIMENT.md            # Paper experiment reproduction guide
├── environment.yml                      # Exported environment snapshot
├── requirements.txt                     # Python requirements
├── setup.sh                             # Environment setup script
├── run_pipeline.sh                      # Master pipeline runner
├── clean.sh                             # Utility cleaner
├── 0_ragas_evaluation.py                # Deprecated legacy entry point
├── 1_neo4j_base_importer.py
├── 1.5_gwas_context_importer.py
├── 2_go_importer.py
├── 3_hpo_importer.py
├── 4_reactome_importer.py
├── 5_clinvar_importer.py
├── 6_pubmed_fetcher.py
├── 7_ner_importer.py
├── 8_ner_reconciliation.py
├── 9_create_embeddings.py
├── 10_dgidb_importer.py
├── 11_ppi_importer.py
├── 12_encode_importer.py
├── 13_clean_gene_nodes.py
├── 14_opentargets_importer.py
├── api.py
├── query_engine.py
├── test_local.py                        # Deprecated legacy entry point
└── evaluation/                          # Canonical paper experiment pipeline
```

**Note:** The GWAS VCF and LLM model files are large (2-17 GB) and are not included in the repository. They must be downloaded separately as described in the setup instructions.

---

## 1. Setup Instructions (Ubuntu 20.04/22.04/24.04)

### 1.1 Clone the Repository

```bash
cd ~/Desktop
git clone https://github.com/<your-org>/Integrative-Genomic-Knowledge-Framework-IGKF.git \
    Integrative-Genomic-Knowledge-Framework-IGKF-main
cd Integrative-Genomic-Knowledge-Framework-IGKF-main
```

### 1.2 Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    wget curl git \
    bcftools \
    build-essential \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libffi-dev
```

**Key system requirements:**
- `bcftools` - VCF filtering and manipulation
- `build-essential` - C/C++ compilers for Python extensions
- `libssl-dev`, `libffi-dev` - Required for cryptography packages

### 1.3 Install Miniconda (if not already installed)

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
# Accept license, install to /home/<user>/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

Restart your shell or run:
```bash
source ~/.bashrc
```

Then navigate to the project:
```bash
cd ~/Desktop/Integrative-Genomic-Knowledge-Framework-IGKF-main
```

### 1.4 Create and Populate the gwas-env Environment

The project uses a single conda environment named `gwas-env` with carefully versioned packages.

**Automated setup** (recommended):

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Remove any existing `gwas-env` environment
2. Create fresh Python 3.10 environment
3. Install core scientific packages (NumPy, Pandas, SciPy) via conda
4. Install bioinformatics tools (CrossMap, bx-python, pyBigWig) via conda
5. Install all Python packages via pip
6. Install the SciSpaCy large language model
7. Validate the installation

**Manual activation** (for subsequent sessions):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gwas-env
cd ~/Desktop/Integrative-Genomic-Knowledge-Framework-IGKF-main
```

### 1.5 Verify GPU Support (Optional but Recommended)

Check if PyTorch can detect your GPU:

```bash
conda activate gwas-env
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

If CUDA is available, the system will use GPU acceleration for embeddings and model inference.

---

## 2. Neo4j via Docker (Community Edition 5.18.0)

This project runs Neo4j Community Edition via Docker with APOC and neosemantics (n10s) plugins.

### 2.1 Install Docker & Docker Compose

**Quick installation script:**

```bash
chmod +x docker.sh
./docker.sh
```

Or follow the official Docker documentation for Ubuntu.

**Verify installation:**

```bash
docker run --rm hello-world
docker compose version
```

### 2.2 Start Neo4j

The `docker-compose.yml` is already configured with:
- Neo4j Community 5.18.0
- APOC and n10s plugins
- 4GB initial heap, 8GB max heap
- 4GB page cache
- Ports 7474 (HTTP) and 7687 (Bolt)

**Create required directories:**

```bash
mkdir -p neo4j/data neo4j/logs neo4j/import neo4j/plugins backups
```

**Start Neo4j:**

```bash
docker compose up -d neo4j
```

**Check status:**

```bash
docker ps
```

**Access Neo4j Browser:**

- **URL:** http://localhost:7474
- **Username:** `neo4j`
- **Password:** `password`

**Important:** If you change the password in `docker-compose.yml`, update the connection settings in all Python scripts accordingly (or use environment variables).

---

## 3. Variant Effect Predictor (VEP) Setup

The pipeline uses Ensembl VEP via Docker for variant annotation.

### 3.1 Install VEP Cache

Follow the official Ensembl documentation:
https://useast.ensembl.org/info/docs/tools/vep/script/vep_download.html#docker

**Quick setup:**

```bash
# Create VEP data directory
mkdir -p ~/vep_data

# Download and install VEP cache for GRCh38
docker run --rm -v ~/vep_data:/data ensemblorg/ensembl-vep \
    perl INSTALL.pl -a cf -s homo_sapiens -y GRCh38 -c /data --CONVERT
```

This will download approximately 15-20 GB of data. The cache is reusable across runs.

---

## 3.5. Local LLM Model Setup

The canonical paper experiment uses **Qwen3-8B** as the only local generation model. Qwen3 generation must run in verified non-thinking mode. The evaluation code applies the Qwen chat template with `enable_thinking=False` before sending prompts to `llama-cpp-python`; it fails loudly if that template control cannot be verified.

### 3.5.1 Download the Model

**Model Repository:** https://huggingface.co/Qwen/Qwen3-8B-GGUF

**Required model:**
- **Filename:** `Qwen3-8B-Q4_K_M.gguf`
- **Quantization:** Q4_K_M
- **Native context:** 32,768 tokens

Download via `huggingface-cli`:

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

cd ~/Desktop/Integrative-Genomic-Knowledge-Framework-IGKF-main
huggingface-cli download Qwen/Qwen3-8B-GGUF \
    Qwen3-8B-Q4_K_M.gguf \
    --local-dir models/
```

### 3.5.2 Verify Model Installation

```bash
cd ~/Desktop/Integrative-Genomic-Knowledge-Framework-IGKF-main
ls -lh models/Qwen3-8B-Q4_K_M.gguf
```

### 3.5.3 Dry Run the Paper Evaluation Setup

```bash
conda activate gwas-env
python -m evaluation.run_generations --dry-run
```

### 3.5.4 Hardware Requirements

**Minimum (CPU only):**
- 32 GB RAM
- 20 GB free disk space
- Inference: ~2-5 tokens/second

**Recommended (GPU):**
- 12 GB GPU VRAM or more
- 32 GB system RAM
- 20 GB free disk space
- Inference: ~20-50 tokens/second

**Hardware parameters for the paper experiment are centralized in `evaluation/configs/paper_experiment.yaml`.**
Do not reduce context length, GPU offload, batch size, model file, or generation parameters ad hoc for paper runs; any change must be explicit in that configuration and recorded in the run metadata.

---

## 4. Running the Pipeline

### 4.1 Prerequisites Checklist

Before running the pipeline, ensure:
- ✅ `gwas-env` conda environment is active
- ✅ Neo4j Docker container is running (`docker ps` shows neo4j)
- ✅ VEP cache is installed in `~/vep_data`
- ✅ Local LLM model downloaded to `models/` directory
- ✅ You are in the project root directory
- ✅ You have downloaded or will download the GWAS data

**Note:** The graph construction pipeline itself does not require the local LLM model, but querying the system and running Qwen3 generation require the configured Qwen3-8B GGUF file.

### 4.2 Master Pipeline Script

Make the pipeline script executable:

```bash
chmod +x run_pipeline.sh
```

Run the complete pipeline:

```bash
./run_pipeline.sh
```

**What the pipeline does:**

1. **[Steps 0-2]** Data preprocessing:
   - Creates necessary directories
   - Downloads GWAS VCF for `ukb-d-2395_1`
   - Filters for genome-wide significant variants (LP > 7.3)
   - Performs liftover from hg19/hg37 to hg38
   - Cleans and normalizes VCF format

2. **[Step 3]** Variant annotation:
   - Runs VEP with GO plugin
   - Annotates with gene symbols, consequences, GO terms, UniProt IDs
   - Uses 2kb upstream/downstream windows for regulatory regions

3. **[Steps 4-5]** Base graph construction:
   - Loads mutations and genes into Neo4j
   - Creates GWAS study context node
   - Establishes `AFFECTS` relationships

4. **[Steps 6-12, 14]** Ontology and database integration:
   - Imports Gene Ontology (GO) with all relationship types
   - Imports Human Phenotype Ontology (HPO)
   - Imports Reactome pathways
   - Imports ClinVar clinical annotations
   - Imports DGIdb drug-gene interactions
   - Imports STRING PPI data
   - Imports ENCODE V4 cCRE regulatory elements and variant-overlap links
   - Imports Open Targets disease associations and drug evidence (via GraphQL API)

   Note: the current validated graph uses cCRE genomic overlap, accession, evidence
   accession, and cCRE class labels. Biosample-specific ENCODE activity fields such
   as `active_in` are not part of the current experiment configuration and are not
   queried during GraphRAG retrieval.

5. **[Step 13]** Data cleaning:
   - Removes invalid gene nodes (malformed symbols)
   - Ensures graph integrity

6. **[Steps 13-15]** Literature enrichment:
   - Fetches PubMed abstracts for relevant genes
   - Runs Named Entity Recognition (NER)
   - Reconciles entities with graph nodes

7. **[Final step]** Vector database:
   - Creates embeddings using S-PubMedBert-MS-MARCO
   - Stores in ChromaDB for semantic search

**Expected runtime:** 2-6 hours depending on hardware (GPU significantly speeds up embedding creation)

### 4.3 Monitoring Progress

The pipeline provides detailed logging. To monitor in real-time:

```bash
tail -f *.log
```

Individual scripts can be run separately if a step fails:

```bash
conda activate gwas-env
python 1_neo4j_base_importer.py  # Run specific importer
```

---

## 5. Querying the System (GraphRAG)

Once the pipeline completes successfully, you can query the integrated knowledge graph.

**Prerequisites for querying:**
- ✅ Pipeline has completed successfully
- ✅ Neo4j contains the knowledge graph
- ✅ ChromaDB contains embeddings
- ✅ Local LLM model is downloaded (see Section 3.5)

### 5.1 Interactive Query Engine

Start the interactive query interface:

```bash
conda activate gwas-env
python query_engine.py
```

The query engine performs hybrid retrieval:
1. **Graph retrieval:** Executes Cypher queries on Neo4j
2. **Vector retrieval:** Semantic search on ChromaDB embeddings
3. **Context fusion:** Combines both contexts
4. **Answer generation:** Uses Qwen3-8B locally in verified non-thinking mode, or the configured OpenAI model for the proprietary comparison

Regulatory-element context is limited to the validated graph fields currently
loaded by `12_encode_importer.py`: cCRE accession, evidence accession, class label,
coordinates, and overlapping GWAS variants/genes. It does not assume
biosample-specific cCRE activity annotations.

**Example queries:**

```
This knowledge graph was generated from a GWAS study. What are the top 
genes associated with the phenotype and what biological pathways are they 
involved in?
```

```
Explain the biological mechanism connecting the AR gene to the phenotype 
studied. Include relevant pathways and protein interactions.
```

### 5.2 API Server

Start the REST API server:

```bash
conda activate gwas-env
python api.py
```

The API exposes endpoints for programmatic access to the RAG system.

---

## 6. Canonical Paper Evaluation

The paper experiment is run through the scripts in `evaluation/`. Generation, OpenAI Batch management, judging, retrieval evaluation, statistical analysis, tables, and figures are intentionally separate steps.

For the exact end-to-end reproduction checklist, see `REPRODUCING_EXPERIMENT.md`.

The preferred publication CLI wrapper is:

```bash
python -m evaluation.cli --help
```

It delegates to the same underlying scripts documented below.

### 6.1 Conditions

The canonical configuration is `evaluation/configs/paper_experiment.yaml` and defines exactly four generation conditions:

- `qwen3_8b_baseline`
- `qwen3_8b_igkf`
- `gpt_5_4_mini_baseline`
- `gpt_5_4_mini_igkf`

Qwen3-8B is the only local generation model. It is configured through llama.cpp using the Qwen tokenizer chat template with `enable_thinking=False`; generation fails if that non-thinking template cannot be verified.

The proprietary generator is configured as `gpt-5.4-mini-2026-03-17` with reasoning effort `none`. The paper judge is configured as `gpt-5.5` with reasoning effort `high`; OpenAI may resolve this alias to a dated model snapshot in returned Batch metadata.

The final annotated benchmark is `data/testset.jsonl` and is expected to contain exactly 205 records with stable `question_id`, `question`, `category`, `answerability`, `expected_behavior`, and `manual_review_required` fields. Benchmark metadata are used for stratification and analysis only; generation prompts receive the question text and the appropriate baseline or IGKF context.

### 6.2 Dry Run

The dry run validates configuration, question loading, condition expansion, output paths, metadata creation, and Qwen3 non-thinking chat-template support without calling OpenAI APIs or loading the full GGUF model.

```bash
conda activate gwas-env
python -m evaluation.cli dry-run
```

### 6.3 Benchmark Validation

```bash
conda activate gwas-env
python -m evaluation.cli validate-benchmark --force
```

Validation fails loudly if the benchmark does not contain exactly 205 usable questions or if IDs, question text, schema, or annotation fields are malformed.

### 6.4 Local Qwen3 Generation

```bash
conda activate gwas-env

python -m evaluation.cli generate-local --conditions qwen3_8b_baseline
python -m evaluation.cli generate-local --conditions qwen3_8b_igkf
```

Useful development flags:

```bash
python -m evaluation.cli generate-local --conditions qwen3_8b_baseline --limit 3
python -m evaluation.cli generate-local --conditions qwen3_8b_baseline --question-ids Q001,Q002
python -m evaluation.cli generate-local --conditions qwen3_8b_baseline --force --question-ids Q001
```

Raw generations are written incrementally under `evaluation/outputs/raw_generations/` and are not overwritten unless `--force` is supplied.

### 6.5 GPT-5.4 Mini Batch Generation

OpenAI Batch jobs require `OPENAI_API_KEY` in the shell environment. Preparing Batch input files does not itself submit paid requests.

```bash
conda activate gwas-env
export OPENAI_API_KEY="..."

python -m evaluation.cli prepare-gpt-batch \
  --conditions gpt_5_4_mini_baseline,gpt_5_4_mini_igkf

# Inspect the printed manifest first, then submit:
python -m evaluation.cli batch submit evaluation/outputs/batch_metadata/<generation_manifest>.json
python -m evaluation.cli batch status evaluation/outputs/batch_metadata/<generation_manifest>.json
python -m evaluation.cli batch download evaluation/outputs/batch_metadata/<generation_manifest>.json
python -m evaluation.cli batch normalize-generations evaluation/outputs/batch_metadata/<generation_manifest>.json
```

The generation Batch manifest records request count, estimated input tokens, output-token budget, file ID, Batch ID, status, output file ID, and raw downloaded JSONL path.

### 6.6 GPT-5.5 Batch Judging

Judging is separate from generation. Pass A scores all four anonymized answers per question for general answer quality. Pass B scores only IGKF answers against the exact IGKF context supplied to generation.

```bash
conda activate gwas-env
export OPENAI_API_KEY="..."

python -m evaluation.cli prepare-judge-batch --pass-name judge_pass_a_general_quality
python -m evaluation.cli batch submit evaluation/outputs/batch_metadata/<pass_a_manifest>.json
python -m evaluation.cli batch status evaluation/outputs/batch_metadata/<pass_a_manifest>.json
python -m evaluation.cli batch download evaluation/outputs/batch_metadata/<pass_a_manifest>.json
python -m evaluation.cli batch normalize-judges evaluation/outputs/batch_metadata/<pass_a_manifest>.json --force

python -m evaluation.cli prepare-judge-batch --pass-name judge_pass_b_igkf_specific
python -m evaluation.cli batch submit evaluation/outputs/batch_metadata/<pass_b_manifest>.json
python -m evaluation.cli batch status evaluation/outputs/batch_metadata/<pass_b_manifest>.json
python -m evaluation.cli batch download evaluation/outputs/batch_metadata/<pass_b_manifest>.json
python -m evaluation.cli batch normalize-judges evaluation/outputs/batch_metadata/<pass_b_manifest>.json --force
```

The current configuration uses `openai.judge_max_output_tokens` for judge batches. This is intentionally separate from the GPT-5.4 mini generation budget so judge retries can use a larger output cap without changing generation assumptions.

### 6.7 Statistics, Tables, and Figures

```bash
conda activate gwas-env
python -m evaluation.cli analyze --force-manifest
```

The judge scores answer relevance, biomedical factual correctness, completeness, and uncertainty/abstention for all conditions. Context groundedness and context utilization are scored only for IGKF conditions using the exact context supplied to generation. Retrieval metrics are only calculated when independent manual relevance judgments are available.

The statistical analysis uses paired Wilcoxon signed-rank tests for the within-model IGKF comparisons and applies Holm correction across the primary tests. Outputs are machine-readable CSV files under `evaluation/outputs/statistics/`.

Publication-oriented figures are generated from `evaluation/outputs/statistics/judge_scores_long.csv` with deterministic bootstrap 95% confidence intervals and are written as PNG, SVG, and PDF files under `evaluation/outputs/figures/`.

For the post-hoc convergence and ceiling-effect diagnostics, run:

```bash
conda activate gwas-env
python -m evaluation.cli analyze-extended
```

This reads the existing normalized judge scores only. It writes score distributions, ceiling proportions, paired question-level deltas, four planned paired comparisons with Holm correction, a documented nonparametric fallback for the 2 x 2 model-by-GraphRAG analysis, and publication figures under `evaluation/outputs/extended_analysis/`.

For the secondary blinded pairwise comparison between the two GraphRAG systems, prepare and inspect the bidirectional Batch job before submitting:

```bash
conda activate gwas-env
python -m evaluation.cli prepare-pairwise-judge --dry-run
python -m evaluation.cli prepare-pairwise-judge
```

This compares only `qwen3_8b_igkf` and `gpt_5_4_mini_igkf`, randomizes A/B assignment by question with a fixed seed, creates a reversed-orientation request for position-bias control, and stores the hidden mapping separately from the judge prompts. Do not submit the manifest until the request count and cost estimate have been reviewed.

After an approved Batch run is downloaded, normalize and analyze the pairwise results with:

```bash
python -m evaluation.cli batch normalize-pairwise evaluation/outputs/batch_metadata/<pairwise_manifest>.json
```

The completed pairwise analysis used GPT-5.5 as the judge and compared only the two GraphRAG conditions. Final reconciled outcomes were:

- Qwen + GraphRAG wins: 23/205 (11.2%)
- GPT-5.4 mini + GraphRAG wins: 165/205 (80.5%)
- Ties: 8/205 (3.9%)
- Position-unstable cases: 9/205 (4.4%)
- Parse/incomplete cases after targeted retries: 0

Among decisive non-unstable comparisons, Qwen's win rate was 0.122 (95% CI 0.079-0.178; exact binomial p = 1.14e-27). This secondary analysis indicates that equal median absolute scores do not establish model equivalence; the direct blinded pairwise judge still distinguished the two GraphRAG systems under this benchmark and judge.

Paper-facing pairwise outputs are copied to:

- `evaluation/outputs/tables/final_pairwise_graphrag_qwen_vs_mini_pairwise_summary.csv`
- `evaluation/outputs/tables/final_pairwise_graphrag_qwen_vs_mini_pairwise_results.csv`
- `evaluation/outputs/tables/final_pairwise_graphrag_qwen_vs_mini_pairwise_statistics.json`
- `evaluation/outputs/figures/figure_5_pairwise_graphrag_outcome_proportions.*`
- `evaluation/outputs/figures/figure_6_pairwise_graphrag_decisive_win_share.*`

The actual recorded Batch usage across the initial run and targeted retries was 1,143,611 input tokens and 119,835 output tokens, with an estimated Batch cost of about $4.66.

### 6.8 Output Layout

```text
evaluation/outputs/
├── raw_generations/        # One JSON file per condition/question
├── retrieval_packages/     # Saved IGKF retrieval context packages
├── batch_inputs/           # OpenAI Batch request JSONL files
├── batch_metadata/         # Batch manifests, mappings, normalization reports
├── batch_outputs/          # Raw downloaded OpenAI Batch JSONL outputs
├── judge_scores/           # Normalized per-condition judge score JSON files
├── statistics/             # Long-form scores, summaries, Wilcoxon tests
├── tables/                 # Paper-facing CSV tables
├── figures/                # PNG/SVG/PDF figures with 95% bootstrap CIs
├── extended_analysis/      # Ceiling, distribution, paired-delta diagnostics
├── pairwise_judge/         # Blinded pairwise GraphRAG judge outputs
└── metadata/               # Benchmark validation, reproducibility, and publication manifests
```

---

## 7. Deprecated Evaluation Entry Points

`0_ragas_evaluation.py` and `test_local.py` are retained as guarded compatibility entry points only. They no longer run the paper experiment because the old flow mixed generation, judging, and ambiguous faithfulness scoring in one place.

---

## 8. Cleaning Up

### 8.1 Stop Neo4j

```bash
docker compose down
```

### 8.2 Clean Generated Files

```bash
chmod +x clean.sh
./clean.sh --yes
```

This removes:
- Neo4j data directory
- ChromaDB vector store
- PubMed abstract cache
- Log files
- Selected legacy temporary scripts

It does not remove `evaluation/outputs/` by default. Do not delete paper experiment outputs unless you have intentionally archived them.

### 8.3 Remove Conda Environment

```bash
conda deactivate
conda env remove -n gwas-env
```

---

## 9. Troubleshooting

### 9.1 GPU Not Detected

**Check PyTorch CUDA:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False:**
- Verify NVIDIA drivers: `nvidia-smi`
- Check CUDA version compatibility
- Reinstall PyTorch with correct CUDA version

### 9.2 Neo4j Connection Errors

**Verify container is running:**
```bash
docker ps | grep neo4j
```

**Check logs:**
```bash
docker logs neo4j
```

**Common issues:**
- Port 7687 already in use
- Insufficient permissions on `neo4j/data` directory
- Memory settings too high for system

**Fix permissions:**
```bash
sudo chown -R 7474:7474 neo4j/data
```

### 9.3 Out of Memory Errors

**For Neo4j:**
- Reduce heap size in `docker-compose.yml`
- Use smaller page cache

**For Python/PyTorch:**
- Reduce batch sizes in scripts
- Use CPU instead of GPU for embeddings
- Process data in smaller chunks

### 9.4 VEP Annotation Fails

**Common causes:**
- VEP cache not downloaded
- Incorrect VEP cache path in `run_pipeline.sh`
- Insufficient disk space

**Check VEP cache:**
```bash
ls -lh ~/vep_data/
```

### 9.5 ChromaDB Errors

**Pydantic version conflicts:**
The environment includes `pydantic-settings` to handle ChromaDB 1.3.4 compatibility with Pydantic v2.

**If you still see BaseSettings errors:**
```bash
pip install pydantic-settings
```

### 9.6 spaCy Model Issues

**Download model manually:**
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

**Validate installation:**
```bash
python -m spacy validate
```

### 9.7 bcftools Not Found

**Install:**
```bash
sudo apt-get install -y bcftools
```

**Verify:**
```bash
bcftools --version
```

### 9.8 Figure Generation Fails

`evaluation.make_figures` requires Matplotlib in `gwas-env`.

```bash
conda activate gwas-env
python -m pip install matplotlib==3.10.9
python -m evaluation.make_figures
```

The script reads saved judge scores only; it does not call OpenAI APIs.

---

## 10. Performance Optimization

### 10.1 Neo4j Tuning

Edit `docker-compose.yml` to adjust memory based on your system:

**For 16GB RAM systems:**
```yaml
NEO4J_dbms_memory_heap_initial__size=2G
NEO4J_dbms_memory_heap_max__size=4G
NEO4J_dbms_memory_pagecache_size=2G
```

**For 32GB+ RAM systems:**
```yaml
NEO4J_dbms_memory_heap_initial__size=4G
NEO4J_dbms_memory_heap_max__size=8G
NEO4J_dbms_memory_pagecache_size=4G
```

### 10.2 Embedding Generation

**Use GPU (automatic if available):**
- 10-50x faster than CPU
- Requires CUDA-compatible GPU
- No code changes needed

**Batch size tuning:**
Edit `9_create_embeddings.py` to adjust batch sizes based on GPU memory.

### 10.3 Pipeline Parallelization

Most importers can run in parallel after the base graph is built:
- Steps 2-5, 10-12, and 14 can run concurrently
- Use GNU parallel or background jobs

---

## 11. Citation

If you use this framework in your research, please cite:

```bibtex
@software{igkf2025,
  title = {Integrative Genomic Knowledge Framework},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/your-org/IGKF}
}
```

---

## 12. License

See `LICENSE` file for details.

---

## 13. Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide system information, error logs, and steps to reproduce

---

## Appendix A: Package Version Summary

### Critical Version Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | Required for compatibility |
| PyTorch | 2.6.0 | CUDA 12.4 build in the tested environment |
| spaCy | 3.7.4 | Required by SciSpaCy 0.5.5 |
| SciSpaCy | 0.5.5 | NER for biomedical text |
| Neo4j | 5.18.0 | Community Edition via Docker |
| ChromaDB | 1.3.4 | Vector store |
| RAGAS | 0.4.3 | Installed for legacy compatibility; not the canonical paper judge |
| LangChain | 0.3.27 | Pipeline compatibility |
| Sentence Transformers | 5.1.2 | Embedding models |
| Transformers | 4.57.1 | HuggingFace models |
| NumPy | 1.26.4 | Core scientific computing |
| Pandas | 2.0.3 | Data manipulation |
| Matplotlib | 3.10.9 | Paper figures |
| PyYAML | 6.0.3 | Experiment configuration loading |

### Bioinformatics Tools

| Package | Source | Purpose |
|---------|--------|---------|
| bcftools | apt | VCF manipulation |
| CrossMap | bioconda | Genome liftover |
| bx-python | bioconda | Genome intervals |
| pyBigWig | bioconda | BigWig file handling |
| pysam | bioconda | SAM/BAM/VCF parsing |

### LangChain Ecosystem

The tested environment uses the LangChain 0.3 series:
- `langchain==0.3.27`
- `langchain-core==0.3.80`
- `langchain-community==0.3.31`
- `langchain-openai==0.3.35`

---

**Last Updated:** July 2026  
**Environment:** Ubuntu 24.04, Python 3.10, CUDA-capable local workstation
