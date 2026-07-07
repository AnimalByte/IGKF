import os
import logging
import pandas as pd
import time

try:
    import pronto
except ImportError:
    print("The 'pronto' library is required for this script.")
    print("Please install it by running: pip install pronto")
    exit()

from config import get_neo4j_driver, download_file, DOWNLOAD_DIR

# --- HPO Import Functions ---
def import_hpo_ontology_manual(driver, hpo_obo_path):
    """
    Manually parses the HPO OBO file and imports terms and relationships into Neo4j.
    This is a more robust alternative to the n10s procedure.
    """
    if not os.path.exists(hpo_obo_path):
        logging.warning(f"HPO file not found at {hpo_obo_path}. Skipping ontology import.")
        return False

    logging.info("Parsing HPO OBO file with 'pronto' library...")
    try:
        ontology = pronto.Ontology(hpo_obo_path)
    except Exception as e:
        logging.error(f"Failed to parse OBO file with pronto: {e}")
        return False

    terms_to_import = []
    for term in ontology.terms():
        if term.id.startswith("HP:"):
            term_data = {
                "hpo_id": term.id,
                "name": term.name,
                "definition": term.definition if term.definition else "No definition available."
            }
            terms_to_import.append(term_data)

    logging.info(f"Found {len(terms_to_import)} HPO terms to import.")

    # Import nodes in batches
    with driver.session() as session:
        session.run("CREATE INDEX hpo_id_index IF NOT EXISTS FOR (n:Phenotype) ON (n.hpo_id)").consume()

        node_query = """
        UNWIND $rows AS row
        MERGE (p:Phenotype {hpo_id: row.hpo_id})
        SET p.name = row.name,
            p.definition = row.definition
        """
        batch_size = 5000
        logging.info("Importing Phenotype nodes...")
        for i in range(0, len(terms_to_import), batch_size):
            batch = terms_to_import[i:i + batch_size]
            session.run(node_query, rows=batch).consume()
            logging.info(f"Imported node batch {i // batch_size + 1}/{(len(terms_to_import) + batch_size - 1) // batch_size}...")

    # Import relationships
    rels_to_import = []
    for term in ontology.terms():
        if term.id.startswith("HP:"):
            for superclass in term.superclasses(distance=1):
                if superclass.id.startswith("HP:"):
                    rels_to_import.append({"subclass_id": term.id, "superclass_id": superclass.id})

    logging.info(f"Found {len(rels_to_import)} IS_A relationships to import.")

    with driver.session() as session:
        rel_query = """
        UNWIND $rows AS row
        MATCH (sub:Phenotype {hpo_id: row.subclass_id})
        MATCH (sup:Phenotype {hpo_id: row.superclass_id})
        MERGE (sub)-[:IS_A]->(sup)
        """
        batch_size = 10000
        logging.info("Importing IS_A relationships...")
        for i in range(0, len(rels_to_import), batch_size):
            batch = rels_to_import[i:i + batch_size]
            result = session.run(rel_query, rows=batch)
            summary = result.consume()
            logging.info(f"Imported relationship batch {i // batch_size + 1}/{(len(rels_to_import) + batch_size - 1) // batch_size}. Relationships created: {summary.counters.relationships_created}")

    # Verify that the import was successful by counting the nodes.
    with driver.session() as session:
        result = session.run("MATCH (p:Phenotype) RETURN count(p) as count")
        count = result.single()['count']
        logging.info(f"Verification: Found {count} nodes with the :Phenotype label.")
        return count > 0

def import_gene_phenotype_associations(driver, associations_path):
    """Reads the HPO annotation file and creates relationships between Genes and Phenotypes."""
    if not os.path.exists(associations_path):
        logging.warning(f"HPO association file not found at {associations_path}. Skipping.")
        return

    logging.info(f"Parsing gene-phenotype association file: {associations_path}")
    try:
        df = pd.read_csv(associations_path, sep='\t', comment='#', header=0)
        df.columns = [col.replace('#', '').strip() for col in df.columns]
        logging.info(f"Successfully parsed {len(df)} associations.")

        records = df[['gene_symbol', 'hpo_id']].drop_duplicates().to_dict('records')
        logging.info(f"Found {len(records)} unique gene-phenotype associations to import.")

        query = """
        UNWIND $rows AS row
        MATCH (g:Gene {symbol: row.gene_symbol})
        MATCH (p:Phenotype {hpo_id: row.hpo_id})
        MERGE (g)-[:ASSOCIATED_WITH]->(p)
        """

        batch_size = 50000
        with driver.session() as session:
            session.run("CREATE INDEX gene_symbol_index IF NOT EXISTS FOR (n:Gene) ON (n.symbol)").consume()

            logging.info("Creating relationships between genes and phenotypes...")
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                result = session.run(query, rows=batch)
                summary = result.consume()
                logging.info(f"Imported relationship batch {i // batch_size + 1}/{(len(records) + batch_size - 1) // batch_size}. Relationships created: {summary.counters.relationships_created}")

        logging.info("Gene-phenotype relationship import complete.")

    except Exception as e:
        logging.error(f"Error processing HPO association file: {e}")

# --- Main execution block ---
def main_hpo_pipeline():
    """Orchestrates the download and import of HPO data."""
    try:
        logging.info("--- Starting HPO Data Import Pipeline ---")

        time.sleep(5)

        hpo_obo_url = "http://purl.obolibrary.org/obo/hp.obo"
        phenotype_to_genes_url = "http://purl.obolibrary.org/obo/hp/hpoa/phenotype_to_genes.txt"

        hpo_obo_path = download_file(hpo_obo_url, DOWNLOAD_DIR, "hp.obo")
        associations_path = download_file(phenotype_to_genes_url, DOWNLOAD_DIR, "phenotype_to_genes.txt")

        driver = get_neo4j_driver()

        with driver.session() as session:
            logging.info("Clearing any previous HPO data...")
            session.run("MATCH (n:Phenotype) DETACH DELETE n").consume()

        labeling_successful = import_hpo_ontology_manual(driver, hpo_obo_path)

        if labeling_successful:
            import_gene_phenotype_associations(driver, associations_path)
        else:
            logging.error("Phenotype node creation failed. Aborting gene-phenotype association import.")

        driver.close()
        logging.info("--- HPO Data Import Pipeline Finished Successfully! ---")

    except Exception as e:
        logging.error(f"An error occurred in the HPO pipeline: {e}")

if __name__ == "__main__":
    main_hpo_pipeline()










