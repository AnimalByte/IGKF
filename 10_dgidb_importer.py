import os
import logging
import pandas as pd
import time
from config import get_neo4j_driver, download_file, DOWNLOAD_DIR

def import_dgidb_data(driver, tsv_path):
    """
    Parses the DGIdb data and creates :Drug nodes and :INTERACTS_WITH relationships
    to existing :Gene nodes in the graph.
    """
    if not os.path.exists(tsv_path):
        logging.error(f"DGIdb data file not found at {tsv_path}. Cannot proceed.")
        return

    logging.info("Parsing DGIdb data...")
    try:
        # The file is tab-separated
        df = pd.read_csv(tsv_path, sep='\t')
        
        # We are interested in linking drug names to gene names
        records = df[['drug_name', 'gene_name']].dropna().drop_duplicates().to_dict('records')
        logging.info(f"Found {len(records)} unique drug-gene interactions in DGIdb.")

        # This query will create Drug nodes and link them to existing Gene nodes
        query = """
        UNWIND $rows AS row
        // Find the existing Gene node by its symbol
        MATCH (g:Gene {symbol: row.gene_name})
        
        // Create or merge the Drug node (convert to uppercase for consistency)
        MERGE (d:Drug {name: toUpper(row.drug_name)})
        
        // Create the relationship between the Drug and its target Gene
        MERGE (d)-[:INTERACTS_WITH]->(g)
        """

        with driver.session() as session:
            # Create an index on Drug names for faster lookups
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Drug) ON (n.name)").consume()
            
            logging.info("Importing Drug-Gene interactions into Neo4j...")
            
            batch_size = 50000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                result = session.run(query, rows=batch)
                summary = result.consume()
                nodes_created = summary.counters.nodes_created
                rels_created = summary.counters.relationships_created
                logging.info(f"Imported chunk {i // batch_size + 1}. Nodes created: {nodes_created}, Relationships created: {rels_created}")

        logging.info("DGIdb data import complete.")

    except Exception as e:
        logging.error(f"An error occurred during DGIdb import: {e}")


# --- Main execution block ---
def main_dgidb_pipeline():
    """Orchestrates the download and import of DGIdb data."""
    try:
        logging.info("--- Starting DGIdb Data Import Pipeline ---")
        time.sleep(5)
        
        dgidb_url = "https://www.dgidb.org/data/latest/interactions.tsv"
        
        data_path = download_file(dgidb_url, DOWNLOAD_DIR, "interactions.tsv")

        driver = get_neo4j_driver()
        
        # Clear any previous Drug data for a clean import
        with driver.session() as session:
            logging.info("Clearing any previous Drug data...")
            session.run("MATCH (d:Drug) DETACH DELETE d").consume()
            
        import_dgidb_data(driver, data_path)
        
        driver.close()
        logging.info("--- DGIdb Data Import Pipeline Finished Successfully! ---")

    except Exception as e:
        logging.error(f"An error occurred in the DGIdb pipeline: {e}")

if __name__ == "__main__":
    main_dgidb_pipeline()
