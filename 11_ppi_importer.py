import os
import logging
import gzip
import shutil
import urllib.request
import pandas as pd
import time
from config import get_neo4j_driver, DOWNLOAD_DIR

STRING_VERSION = "12.0"
CONFIDENCE_THRESHOLD = 0

# --- Data Download Function ---
def download_and_decompress(url, directory, filename):
    """Downloads and decompresses a .gz file if the uncompressed version doesn't already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    gz_path = os.path.join(directory, filename)
    uncompressed_path = gz_path.replace('.gz', '')

    if not os.path.exists(uncompressed_path):
        if not os.path.exists(gz_path):
            logging.info(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, gz_path)
            logging.info("Download complete.")

        logging.info(f"Decompressing {filename}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logging.info(f"Decompression complete: {uncompressed_path}")
    else:
        logging.info(f"{uncompressed_path} already exists. Skipping download and decompression.")

    return uncompressed_path

def import_ppi_data(driver, links_path, alias_path):
    """
    Parses STRING-DB data to create protein-protein interaction relationships
    with detailed evidence scores between existing :Gene nodes in the graph.
    """
    if not all([os.path.exists(links_path), os.path.exists(alias_path)]):
        logging.error("One or more STRING data files not found. Cannot proceed.")
        return

    logging.info("Creating map from STRING Protein ID to Gene Symbol...")
    try:
        column_names = ['string_protein_id', 'alias', 'source']
        df_aliases = pd.read_csv(alias_path, sep='\t', header=None, names=column_names, comment='#', low_memory=False)

        hgnc_aliases = df_aliases[df_aliases['source'].str.contains("Ensembl_HGNC_symbol", na=False)]
        string_to_symbol = pd.Series(hgnc_aliases.alias.values, index=hgnc_aliases.string_protein_id).to_dict()
        logging.info(f"Mapped {len(string_to_symbol)} STRING IDs using primary HGNC symbols.")

        remaining_aliases = df_aliases[~df_aliases['string_protein_id'].isin(string_to_symbol.keys())]
        fallback_map = remaining_aliases.groupby('string_protein_id')['alias'].first().to_dict()

        string_to_symbol.update(fallback_map)
        logging.info(f"Mapped a total of {len(string_to_symbol)} STRING IDs after fallback.")

    except Exception as e:
        logging.error(f"Failed to parse alias file: {e}")
        return

    logging.info(f"Processing detailed interaction data...")
    col_names = [
        "protein1", "protein2", "neighborhood", "fusion", "cooccurence",
        "coexpression", "experimental", "database", "textmining", "combined_score"
    ]
    df_links = pd.read_csv(links_path, sep=' ', header=0, names=col_names)

    # Map STRING IDs to Gene Symbols
    df_links['gene1'] = df_links['protein1'].map(string_to_symbol)
    df_links['gene2'] = df_links['protein2'].map(string_to_symbol)

    df_links.dropna(subset=['gene1', 'gene2'], inplace=True)
    df_links = df_links[df_links['gene1'] != df_links['gene2']]

    if CONFIDENCE_THRESHOLD > 0:
        before = len(df_links)
        df_links = df_links[df_links['combined_score'] >= CONFIDENCE_THRESHOLD]
        logging.info(f"Filtered by confidence threshold {CONFIDENCE_THRESHOLD}: {before} → {len(df_links)} interactions.")

    # --- FIX: Filter out non-gene symbols ---
    # Keep only symbols that do not start with a number and are not purely numeric.
    # This prevents creating nodes for things like '7768' or 'A0A024R1R8'.
    df_links = df_links[~df_links['gene1'].astype(str).str.match(r'^\d+$')]
    df_links = df_links[~df_links['gene2'].astype(str).str.match(r'^\d+$')]
    df_links = df_links[~df_links['gene1'].astype(str).str.match(r'^[A-Z0-9]{10}$')]
    df_links = df_links[~df_links['gene2'].astype(str).str.match(r'^[A-Z0-9]{10}$')]
    logging.info("Filtered out interactions with non-standard gene symbols.")
    # --- END FIX ---


    # Normalize interactions to avoid duplicates (e.g., A-B is the same as B-A)
    df_links['sorted_genes'] = df_links.apply(lambda row: tuple(sorted((row['gene1'], row['gene2']))), axis=1)
    df_links = df_links.drop_duplicates(subset=['sorted_genes'])

    interactions = df_links.to_dict('records')
    logging.info(f"Found {len(interactions)} unique interactions with valid gene symbols to import.")

    if not interactions:
        logging.warning("No interactions to import.")
        return

    # This query now sets all individual evidence scores as properties on the relationship
    query = """
    UNWIND $rows AS row
    MERGE (g1:Gene {symbol: row.gene1})
    MERGE (g2:Gene {symbol: row.gene2})
    MERGE (g1)-[r:INTERACTS_WITH]-(g2)
    SET r.source = 'STRING-DB',
        r.neighborhood = toInteger(row.neighborhood),
        r.fusion = toInteger(row.fusion),
        r.cooccurence = toInteger(row.cooccurence),
        r.coexpression = toInteger(row.coexpression),
        r.experimental = toInteger(row.experimental),
        r.database = toInteger(row.database),
        r.textmining = toInteger(row.textmining),
        r.score = toInteger(row.combined_score)
    """

    with driver.session() as session:
        logging.info("Importing detailed PPI relationships into Neo4j...")
        batch_size = 50000
        total_rels_created = 0
        total_nodes_created = 0

        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            result = session.run(query, rows=batch)
            summary = result.consume()

            rels = summary.counters.relationships_created
            nodes = summary.counters.nodes_created
            total_rels_created += rels
            total_nodes_created += nodes

            logging.info(f"Imported chunk {i // batch_size + 1}. Nodes created: {nodes}, Relationships created: {rels}")

        logging.info(f"PPI data import complete. Total nodes created: {total_nodes_created}, Total relationships created: {total_rels_created}.")

# --- Main execution block ---
def main_ppi_pipeline():
    """Orchestrates the download and import of STRING-DB PPI data."""
    try:
        logging.info("--- Starting STRING-DB Protein-Protein Interaction Import Pipeline ---")
        time.sleep(5)

        links_url = f"https://stringdb-static.org/download/protein.links.detailed.v{STRING_VERSION}/9606.protein.links.detailed.v{STRING_VERSION}.txt.gz"
        alias_url = f"https://stringdb-static.org/download/protein.aliases.v{STRING_VERSION}/9606.protein.aliases.v{STRING_VERSION}.txt.gz"

        links_file_path = download_and_decompress(links_url, DOWNLOAD_DIR, f"9606.protein.links.detailed.v{STRING_VERSION}.txt.gz")
        alias_file_path = download_and_decompress(alias_url, DOWNLOAD_DIR, f"9606.protein.aliases.v{STRING_VERSION}.txt.gz")

        driver = get_neo4j_driver()

        # Clear any previous STRING-DB PPI data
        with driver.session() as session:
            logging.info("Clearing any previous STRING-DB PPI relationships...")
            session.run("MATCH ()-[r:INTERACTS_WITH {source: 'STRING-DB'}]-() DELETE r").consume()

        import_ppi_data(driver, links_file_path, alias_file_path)

        driver.close()
        logging.info("--- STRING-DB Import Pipeline Finished Successfully! ---")

    except Exception as e:
        logging.error(f"An error occurred in the PPI pipeline: {e}")

if __name__ == "__main__":
    main_ppi_pipeline()
