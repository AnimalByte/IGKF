#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "  IGKF Environment Setup Script"
echo "  Setting up gwas-env with all dependencies"
echo "=================================================="
echo

# --- Make conda usable inside this script ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
fi

# --- Start fresh ---
echo "[1/5] Removing existing gwas-env (if present)..."
conda deactivate || true
conda env remove -n gwas-env -y || true
echo "✓ Environment cleaned"
echo

# --- Create base environment with Python 3.10 ---
echo "[2/5] Creating gwas-env with Python 3.10..."
conda create -y -n gwas-env python=3.10
conda activate gwas-env
echo "✓ Base environment created"
echo

# --- Install core scientific stack via conda ---
echo "[3/5] Installing core scientific packages via conda..."
conda install -y -c conda-forge \
    numpy=1.26.4 \
    pandas=2.0.3 \
    scipy=1.11.4 \
    pip
echo "✓ Core scientific stack installed"
echo

# --- Install bioinformatics tools via conda ---
echo "[3.5/5] Installing bioinformatics tools (bx-python, CrossMap, pyBigWig)..."
# These need to be built from source via conda as they have C extensions
conda install -y -c conda-forge -c bioconda \
    bx-python \
    crossmap \
    pybigwig \
    pysam
echo "✓ Bioinformatics tools installed"
echo

# --- Install all Python packages via pip ---
echo "[4/5] Installing Python packages via pip..."
pip install --no-cache-dir \
    "biopython==1.86" \
    "pronto==2.7.2" \
    "spacy==3.7.4" \
    "scispacy==0.5.5" \
    "neo4j>=5.18.0,<6.0.0" \
    "chromadb==1.3.4" \
    "sentence-transformers==5.1.2" \
    "torch==2.6.0" \
    "transformers==4.57.1" \
    "scikit-learn==1.7.2" \
    "llama-cpp-python==0.3.33" \
    "langchain==0.3.27" \
    "langchain-core==0.3.80" \
    "langchain-community==0.3.31" \
    "langchain-openai==0.3.35" \
    "langchain-text-splitters==0.3.11" \
    "langsmith==0.4.42" \
    "ragas==0.4.3" \
    "datasets==4.4.1" \
    "openai==2.7.2" \
    "tqdm==4.67.1" \
    "pydantic>=2.7,<3" \
    "pydantic-settings==2.12.0" \
    "python-dotenv==1.2.1" \
    "requests==2.32.5" \
    "fastapi" \
    "matplotlib==3.10.9" \
    "PyYAML==6.0.3" \
    "pre-commit==4.4.0"

echo "✓ Python packages installed"
echo

# --- Install SciSpaCy model ---
echo "[5/5] Installing SciSpaCy large model..."
pip install --no-cache-dir \
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
echo "✓ SciSpaCy model installed"
echo

# --- Validate installation ---
echo "=================================================="
echo "  Validating Installation"
echo "=================================================="
python -c "import spacy; print(f'✓ spaCy {spacy.__version__}')" || echo "✗ spaCy failed"
python -c "import scispacy; print(f'✓ SciSpaCy {scispacy.__version__}')" || echo "✗ SciSpaCy failed"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || echo "✗ PyTorch failed"
python -c "import neo4j; print(f'✓ Neo4j driver installed')" || echo "✗ Neo4j driver failed"
python -c "import chromadb; print(f'✓ ChromaDB {chromadb.__version__}')" || echo "✗ ChromaDB failed"
python -c "import ragas; print(f'✓ RAGAS {ragas.__version__}')" || echo "✗ RAGAS failed"
python -c "import langchain; print(f'✓ LangChain {langchain.__version__}')" || echo "✗ LangChain failed"
python -c "import llama_cpp; print(f'✓ llama-cpp-python {llama_cpp.__version__}')" || echo "✗ llama-cpp-python failed"
python -c "import openai; print(f'✓ OpenAI {openai.__version__}')" || echo "✗ OpenAI failed"
python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" || echo "✗ Matplotlib failed"
python -c "import yaml; print(f'✓ PyYAML {yaml.__version__}')" || echo "✗ PyYAML failed"
echo

# --- Final validation with spacy model ---
echo "Validating spaCy model..."
python -m spacy validate || true
echo

echo "=================================================="
echo "  ✓ Setup Complete!"
echo "=================================================="
echo
echo "Environment 'gwas-env' is ready to use."
echo
echo "To activate in any new shell:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate gwas-env"
echo
echo "GPU Support:"
python -c "import torch; print(f'  PyTorch CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'  GPU device: {torch.cuda.get_device_name(0)}')"
fi
echo
echo "Next steps:"
echo "  1. Start Neo4j: docker compose up -d neo4j"
echo "  2. Download GWAS data to data/ directory"
echo "  3. Run pipeline: ./run_pipeline.sh"
echo "=================================================="
