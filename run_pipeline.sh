#!/bin/bash

# This script runs the entire data processing and import pipeline in the correct order.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Full Data Import and Processing Pipeline ---"

# --- Define Input File ---
INPUT_VCF="data/ukb-d-2395_1.vcf.gz"

# --- Step 0: Setup Directories ---
echo "[0/15] Creating project directories if they do not exist..."
mkdir -p data models results downloads neo4j/data

# --- Initialize Conda in the script's shell ---
echo "Initializing Conda for this script session..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gwas-env

# --- Step 1: Pre-process VCF Data ---
echo "[1/15] Filtering for significant variants (LP > 7.3)..."
bcftools view -i 'LP > 7.3' "$INPUT_VCF" -o results/significant_hg37.vcf

echo "[2/15] Performing LiftOver from hg37 to hg38..."
wget -nc -P downloads/ http://hgdownload.soe.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz
wget -nc -P downloads/ http://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz
gunzip -f downloads/hg19ToHg38.over.chain.gz || true
gunzip -f downloads/hg38.fa.gz || true
echo "Running CrossMap..."
$CONDA_PREFIX/bin/CrossMap vcf downloads/hg19ToHg38.over.chain results/significant_hg37.vcf downloads/hg38.fa results/gwas_hg38.vcf

# Correct chromosome naming and clean the VCF
echo "[2.5/15] Correcting chromosome names and cleaning VCF..."
grep -v "^#" results/gwas_hg38.vcf | cut -f1 | sort -u | awk '{print $1"\tchr"$1}' > results/chr_name_map.txt
bcftools annotate --rename-chrs results/chr_name_map.txt results/gwas_hg38.vcf -O v -o results/gwas_hg38_renamed.vcf
bcftools norm -c s -f downloads/hg38.fa results/gwas_hg38_renamed.vcf -O v -o results/gwas_hg38_cleaned.vcf

# --- Step 2: Annotate with VEP using Docker ---
CORES=$(nproc)
echo "[3/15] Annotating with VEP using $CORES cores..."

# Fix permissions so Docker can write to results directory
chmod -R a+rwx results/

docker run --rm -v $(pwd)/results:/opt/vep/data -v $HOME/vep_data:/opt/vep/.vep ensemblorg/ensembl-vep \
    vep -i /opt/vep/data/gwas_hg38_cleaned.vcf -o /opt/vep/data/egIwc7NRt4hou5yo.txt \
    --cache --dir /opt/vep/.vep --assembly GRCh38 --fork $CORES --force_overwrite --tab \
    --distance 2000 --symbol --biotype --uniprot \
    --fields "Uploaded_variation,Location,Allele,Consequence,IMPACT,SYMBOL,REF_ALLELE,Gene,GO,SWISSPROT,TREMBL" \
    --plugin GO

# --- Step 3: Knowledge Graph Construction ---
echo "[4/15] Building base graph (Mutations -> Genes)..."
python 1_neo4j_base_importer.py

echo "[5/15] Importing GWAS study context..."
python 1.5_gwas_context_importer.py

# --- Run the rest of the importers ---
echo "--- Running Data Importers ---"
python 2_go_importer.py
python 3_hpo_importer.py
python 4_reactome_importer.py
python 5_clinvar_importer.py
python 10_dgidb_importer.py
python 11_ppi_importer.py
python 12_encode_importer.py

echo "--- Importing Open Targets disease associations and drug evidence ---"
python 14_opentargets_importer.py

# --- NEW STEP: Clean the graph of any invalid gene nodes ---
echo "--- Cleaning Invalid Gene Nodes from the Graph ---"
python 13_clean_gene_nodes.py
# --- END NEW STEP ---

# --- Step 4: Unstructured Data Enrichment ---
echo "[13/15] Fetching PubMed abstracts for contextually relevant genes..."
python 6_pubmed_fetcher.py

echo "[14/15] Extracting entities (NER) from abstracts and adding to the graph..."
python 7_ner_importer.py

echo "[15/15] Reconciling extracted entities with known graph nodes..."
python 8_ner_reconciliation.py

echo "--- Building Vector Database ---"
echo "Creating embeddings and storing in ChromaDB..."
python 9_create_embeddings.py

echo "--- Pipeline Complete! ---"
echo "You can now run 'python query_engine.py' to chat with your knowledge graph."
