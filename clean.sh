#!/usr/bin/env bash
set -euo pipefail

# Clean generated graph/build artifacts while preserving paper experiment outputs.
# This script intentionally does not remove evaluation/outputs/.

if [[ "${1:-}" != "--yes" ]]; then
    cat <<'EOF'
Refusing to clean without explicit confirmation.

This removes generated graph/build artifacts, including:
  - __pycache__ directories
  - pubmed_abstracts/
  - chroma_db/
  - backups/
  - neo4j/data
  - root-level temporary archives/logs

It preserves:
  - data/testset.jsonl
  - evaluation/outputs/
  - models/
  - downloads/

Run:
  ./clean.sh --yes
EOF
    exit 2
fi

echo "--- Starting Project Cleanup ---"
echo "Paper experiment outputs under evaluation/outputs/ will be preserved."

echo "Stopping Docker containers if Docker Compose is available..."
if command -v docker >/dev/null 2>&1; then
    docker compose down || docker-compose down || true
fi

echo "Removing generated cache/data directories..."
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
rm -rf pubmed_abstracts/
rm -rf chroma_db/
rm -rf backups/

if [[ -d neo4j/data ]]; then
    echo "Removing neo4j/data using sudo if needed..."
    rm -rf neo4j/data 2>/dev/null || sudo rm -rf neo4j/data
fi

echo "Removing root-level temporary files..."
rm -f ./*.log
rm -f ./*.deb ./*.deb.*
rm -f ./*.zip ./*.tar.gz
rm -f ./*Zone.Identifier

echo "Removing known legacy temporary scripts if present..."
rm -f parser.py
rm -f pathway_extraction.py
rm -f pathway_extraction1.py
rm -f retrival.py
rm -f enrich_genes.py
rm -f debug_uniprot_match.py
rm -f query_engine1.py
rm -f script.sh

echo "Re-creating essential directories..."
mkdir -p results downloads neo4j/data

if command -v sudo >/dev/null 2>&1; then
    echo "Setting permissions for Neo4j data directory..."
    sudo chown -R 7474:7474 neo4j/data || true
fi

echo "--- Cleanup Complete ---"
echo "evaluation/outputs/ was not removed."
