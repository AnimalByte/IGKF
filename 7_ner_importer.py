import os
import logging
import spacy
import time
from config import get_neo4j_driver, ABSTRACTS_DIR, SPACY_MODEL_NAME

# --- Main NER and Import Function ---
def extract_and_load_entities(driver):
    """
    Processes downloaded PubMed abstracts, extracts biomedical entities using a
    scispaCy model, and loads them into the Neo4j graph.
    """
    logging.info("--- Starting NER and Graph Enrichment Pipeline ---")

    # --- Step 1: Load the scispaCy NER Model ---
    logging.info("Checking for GPU and attempting to enable it for spaCy...")
    # --- FIX: Attempt to use GPU for spaCy ---
    if spacy.prefer_gpu():
        logging.info("GPU enabled for spaCy!")
    else:
        logging.warning("GPU not available for spaCy, using CPU. This may be slower.")
    # --- END FIX ---
    
    logging.info(f"Loading NER model: {SPACY_MODEL_NAME}...")
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
        logging.info("NER model loaded successfully.")
    except OSError:
        logging.error(f"Model '{SPACY_MODEL_NAME}' not found. Please run the correct installation commands.")
        logging.error("Run ./setup.sh or install the pinned SciSpaCy model from requirements.txt.")
        return

    # --- Step 2: Prepare Neo4j Indexes ---
    with driver.session() as session:
        logging.info("Creating indexes for new entity types...")
        session.run("CREATE INDEX paper_pmid_index IF NOT EXISTS FOR (n:Paper) ON (n.pmid)").consume()
        session.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)").consume()

    # --- Step 3: Process Abstracts and Load into Graph ---
    logging.info(f"Scanning for abstracts in: '{ABSTRACTS_DIR}'")
    
    query = """
    MERGE (p:Paper {pmid: $pmid})
    SET p.gene_context = $gene_symbol
    
    WITH p
    UNWIND $entities AS entity_data
    
    MERGE (e:Entity {name: entity_data.text})
    WITH p, e, entity_data
    CALL apoc.create.addLabels(e, [entity_data.label]) YIELD node
    
    MERGE (node)-[:MENTIONED_IN]->(p)
    """
    
    total_processed = 0
    for gene_symbol in os.listdir(ABSTRACTS_DIR):
        gene_dir = os.path.join(ABSTRACTS_DIR, gene_symbol)
        if os.path.isdir(gene_dir):
            for filename in os.listdir(gene_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(gene_dir, filename)
                    pmid = os.path.splitext(filename)[0]

                    with open(filepath, 'r', encoding='utf-8') as f:
                        abstract_text = f.read()
                    
                    doc = nlp(abstract_text)
                    
                    entities_to_load = []
                    for ent in doc.ents:
                         if len(ent.text) > 3:
                            entities_to_load.append({
                                "text": ent.text.lower(), 
                                "label": ent.label_.capitalize()
                            })

                    if entities_to_load:
                        unique_entities = [dict(t) for t in {tuple(d.items()) for d in entities_to_load}]
                        with driver.session() as session:
                            session.run(query, pmid=pmid, gene_symbol=gene_symbol, entities=unique_entities)
                    
                    total_processed += 1
                    if total_processed % 100 == 0:
                        logging.info(f"Processed {total_processed} abstracts...")

    logging.info(f"--- NER processing complete! Processed {total_processed} total abstracts. ---")

    # --- Step 4: Final Verification ---
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN count(p) as count").single()
        paper_count = result['count'] if result else 0
        logging.info(f"VERIFICATION: Found {paper_count} :Paper nodes in the database.")
        if paper_count > 0:
            logging.info("SUCCESS: Data was loaded into the graph.")
        else:
            logging.error("FAILURE: No :Paper nodes were created.")

# --- Main execution block ---
def main():
    """Orchestrates the NER pipeline."""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            logging.info("Clearing any previous NER-generated data...")
            session.run("MATCH (n:Paper) DETACH DELETE n").consume()
            session.run("MATCH (e:Entity) WHERE size(labels(e)) = 1 DETACH DELETE e").consume()

        extract_and_load_entities(driver)

        driver.close()

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main()
