import logging
from config import get_neo4j_driver


# --- Main Reconciliation Function ---
def reconcile_entities(driver):
    """
    Finds generic :Entity nodes and merges them with specific, typed nodes
    (:Gene, :Phenotype, :Disease) in smaller, memory-safe batches.
    """
    logging.info("--- Starting Entity Reconciliation Pipeline ---")

    with driver.session() as session:
        # --- Step 1: Fetch all necessary data from the graph ---
        
        logging.info("Fetching generic entities from the graph...")
        entity_result = session.run("MATCH (e:Entity) WHERE size(labels(e)) = 1 RETURN e.name AS name")
        generic_entities = {record['name'] for record in entity_result}
        logging.info(f"Found {len(generic_entities)} generic :Entity nodes to process.")
        
        logging.info("Fetching specific :Phenotype nodes from the graph...")
        phenotype_result = session.run("MATCH (p:Phenotype) RETURN p.name AS name")
        phenotype_map = {name.lower(): name for record in phenotype_result for name in (record['name'] if isinstance(record['name'], list) else [record['name']])}
        logging.info(f"Found {len(phenotype_map)} unique Phenotype names.")

        logging.info("Fetching specific :Disease nodes from the graph...")
        disease_result = session.run("MATCH (d:Disease) RETURN d.name AS name")
        disease_map = {name.lower(): name for record in disease_result for name in (record['name'] if isinstance(record['name'], list) else [record['name']])}
        logging.info(f"Found {len(disease_map)} unique Disease names.")
        
        logging.info("Fetching specific :Gene nodes from the graph...")
        gene_result = session.run("MATCH (g:Gene) RETURN g.symbol AS name")
        gene_map = {record['name'].lower(): record['name'] for record in gene_result}
        logging.info(f"Found {len(gene_map)} unique Gene names.")

        # --- Step 2: Perform reconciliation logic in Python ---

        logging.info("Finding matches between generic entities and specific nodes...")
        phenotype_matches = []
        disease_matches = []
        gene_matches = []

        for entity_name in generic_entities:
            lower_entity_name = entity_name.lower()
            if lower_entity_name in phenotype_map:
                phenotype_matches.append({"entity_name": entity_name, "specific_name": phenotype_map[lower_entity_name]})
            elif lower_entity_name in disease_map:
                disease_matches.append({"entity_name": entity_name, "specific_name": disease_map[lower_entity_name]})
            elif lower_entity_name in gene_map:
                gene_matches.append({"entity_name": entity_name, "specific_name": gene_map[lower_entity_name]})

        logging.info(f"Found {len(phenotype_matches)} Phenotype, {len(disease_matches)} Disease, and {len(gene_matches)} Gene reconciliations to perform.")

        # --- FIX: Use smaller, safer transactions to avoid memory errors ---
        def run_merge_in_batches(matches, specific_label, specific_key, session):
            total_merged = 0
            batch_size = 500  # Smaller batch size for safety
            logging.info(f"Executing merge operations for : {specific_label}...")
            
            merge_query = f"""
            UNWIND $rows AS row
            MATCH (generic:Entity {{name: row.entity_name}})
            MATCH (specific:{specific_label} {{{specific_key}: row.specific_name}})
            WITH generic, specific
            CALL apoc.refactor.mergeNodes([generic, specific], {{properties: 'combine'}}) YIELD node
            RETURN count(node) as merged_count
            """

            for i in range(0, len(matches), batch_size):
                batch = matches[i:i + batch_size]
                result = session.run(merge_query, rows=batch)
                merged_in_batch = sum(record['merged_count'] for record in result)
                total_merged += merged_in_batch
                logging.info(f"Merged batch {i//batch_size + 1}/{-(-len(matches)//batch_size)}. Nodes merged in batch: {merged_in_batch}")
            logging.info(f"Total : {specific_label} nodes merged: {total_merged}")

        run_merge_in_batches(phenotype_matches, "Phenotype", "name", session)
        run_merge_in_batches(disease_matches, "Disease", "name", session)
        run_merge_in_batches(gene_matches, "Gene", "symbol", session)
        # --- END FIX ---

    logging.info(f"--- Entity Reconciliation Complete! ---")


# --- Main execution block ---
def main():
    """Orchestrates the NER pipeline."""
    try:
        driver = get_neo4j_driver()
        
        reconcile_entities(driver)

        driver.close()

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main()
