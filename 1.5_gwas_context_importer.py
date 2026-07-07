import logging
import time
from config import get_neo4j_driver

# --- Study Metadata ---
# This dictionary explicitly defines the context of our GWAS study.
STUDY_METADATA = {
    "trait": "Hair/balding pattern: Pattern 1",
    "ieu_id": "ukb-d-2395_1",
    "source_vcf": "ukb-d-2395_1.vcf.gz",
    "link": "https://gwas.mrcieu.ac.uk/datasets/ukb-d-2395_1/"
}

def import_gwas_context(driver):
    """
    Creates a node for the GWAS study itself and links all existing Mutation
    nodes to it. This provides global context for the dataset.
    """
    logging.info("Importing GWAS study context into the graph...")

    # Query to create the single study node from our metadata dictionary.
    create_study_node_query = """
    MERGE (s:GWAS_Study {ieu_id: $metadata.ieu_id})
    SET s.trait = $metadata.trait,
        s.source_vcf = $metadata.source_vcf,
        s.link = $metadata.link
    RETURN s
    """

    # This query efficiently links all existing mutations to the study node.
    # It uses APOC's periodic iterate for performance and safety on large graphs.
    link_mutations_query = """
    CALL apoc.periodic.iterate(
        "MATCH (m:Mutation) RETURN m",
        "MATCH (s:GWAS_Study {ieu_id: 'ukb-d-2395_1'}) MERGE (m)-[:FROM_STUDY]->(s)",
        {batchSize: 10000, parallel: true}
    )
    """
    # A simpler fallback for environments without APOC.
    fallback_link_query = """
    MATCH (s:GWAS_Study {ieu_id: $ieu_id})
    MATCH (m:Mutation)
    MERGE (m)-[:FROM_STUDY]->(s)
    """

    with driver.session() as session:
        # Step 1: Create the study node
        logging.info(f"Creating/updating GWAS_Study node for trait: '{STUDY_METADATA['trait']}'")
        session.run(create_study_node_query, metadata=STUDY_METADATA).consume()

        # Step 2: Link all mutations
        logging.info("Linking all mutations to the GWAS_Study node...")
        try:
            # Try the high-performance APOC query first.
            session.run(link_mutations_query).consume()
            logging.info("Successfully linked mutations to study node using apoc.periodic.iterate.")
        except Exception:
            # If APOC isn't installed or fails, use the standard, slower query.
            logging.warning("APOC procedure failed. Falling back to standard Cypher. This may be slow.")
            session.run(fallback_link_query, ieu_id=STUDY_METADATA['ieu_id']).consume()
            logging.info("Successfully linked mutations to study node using standard Cypher.")


def main():
    """Main function to run the context import."""
    try:
        logging.info("--- Starting GWAS Study Context Import ---")
        time.sleep(5)
        driver = get_neo4j_driver()
        import_gwas_context(driver)
        driver.close()
        logging.info("--- GWAS Study Context Import Finished Successfully! ---")
    except Exception as e:
        logging.error(f"An error occurred in the context import pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()

