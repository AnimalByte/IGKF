import os
import logging
import pandas as pd
import time
import re
from config import get_neo4j_driver, ANNOTATED_TSV_PATH

def import_base_graph_data(driver, tsv_path):
    """
    Parses the annotated TSV file and loads the base graph of Mutations,
    Genes, and their relationships into Neo4j. This version reads the file
    line-by-line and filters for valid gene symbols.
    """
    if not os.path.exists(tsv_path):
        logging.error(f"Annotated TSV file not found at {tsv_path}. Cannot proceed.")
        return

    logging.info(f"Step 1: Parsing annotated TSV file line-by-line: {tsv_path}")

    header = []
    all_records = []
    skipped_lines = 0

    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('#Uploaded_variation'):
                    header = [h.strip() for h in line.strip().split('\t')]
                    if header and header[0] == '#Uploaded_variation':
                        header[0] = 'Uploaded_variation'
                    continue
                if line.startswith('#') or not line.strip():
                    continue

                fields = [field.strip() for field in line.strip().split('\t')]
                if len(fields) == len(header):
                    record_dict = dict(zip(header, fields))
                    all_records.append(record_dict)
                else:
                    skipped_lines += 1
        
        logging.info(f"File parsing summary: Total lines read: {i+1}, Successfully parsed: {len(all_records)}, Skipped lines: {skipped_lines}")

        # --- FIX: Filter for valid gene symbols before import ---
        # A valid gene symbol should not contain spaces or start with a hyphen.
        # We can use a simple regex for this.
        gene_symbol_pattern = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$')
        
        cleaned_records = []
        for record in all_records:
            symbol = record.get('SYMBOL')
            if symbol and symbol != '-' and gene_symbol_pattern.match(symbol):
                cleaned_records.append(record)
        
        logging.info(f"Filtered records: Kept {len(cleaned_records)} records with valid gene symbols.")
        # --- END FIX ---

        query = """
        UNWIND $rows AS row
        MERGE (m:Mutation {id: row.Uploaded_variation})
        SET m.location = row.Location,
            m.chrom = split(row.Location, ':')[0],
            m.pos = split(split(row.Location, ':')[1], '-')[0],
            m.ref = row.REF_ALLELE,
            m.alt = row.Allele,
            m.consequence = row.Consequence,
            m.impact = row.IMPACT

        WITH m, row
        WHERE row.SYMBOL IS NOT NULL AND row.SYMBOL <> '-' AND row.SYMBOL <> ''
        MERGE (g:Gene {symbol: row.SYMBOL})
        MERGE (m)-[:AFFECTS]->(g)
        """

        with driver.session() as session:
            session.run("CREATE INDEX mutation_id_index IF NOT EXISTS FOR (n:Mutation) ON (n.id)").consume()
            session.run("CREATE INDEX gene_symbol_index IF NOT EXISTS FOR (n:Gene) ON (n.symbol)").consume()

            logging.info("Step 2: Importing Mutations and Genes into Neo4j...")

            chunk_size = 50000
            total_nodes_created = 0
            total_rels_created = 0

            # Use the cleaned_records list for import
            for i in range(0, len(cleaned_records), chunk_size):
                batch = cleaned_records[i:i+chunk_size]
                result = session.run(query, rows=batch)
                summary = result.consume()
                nodes_created = summary.counters.nodes_created
                rels_created = summary.counters.relationships_created
                total_nodes_created += nodes_created
                total_rels_created += rels_created
                logging.info(f"Imported chunk {i // chunk_size + 1}. Nodes created: {nodes_created}, Relationships created: {rels_created}")

        logging.info(f"Base graph import complete. Total nodes created: {total_nodes_created}, Total relationships created: {total_rels_created}.")

    except FileNotFoundError:
        logging.error(f"Annotated file not found at {tsv_path}. Please check the path.")
    except Exception as e:
        logging.error(f"An error occurred during the import process: {e}")

# --- Main execution block ---
def main_pipeline():
    """Orchestrates the base graph import."""
    try:
        logging.info("--- Starting Neo4j Base Import Pipeline ---")

        time.sleep(10)

        driver = get_neo4j_driver()

        import_base_graph_data(driver, ANNOTATED_TSV_PATH)

        driver.close()
        logging.info("--- Neo4j Base Import Pipeline Finished Successfully! ---")

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main_pipeline()
