import os
import logging
import time
import re

try:
    import pronto
except ImportError:
    print("The 'pronto' library is required for this script.")
    print("Please install it by running: pip install pronto")
    exit()

from config import get_neo4j_driver, download_file, ANNOTATED_TSV_PATH, DOWNLOAD_DIR

# --- GO Import Functions ---
def import_go_ontology_manual(driver, go_obo_path):
    """
    Manually parses the GO OBO file and imports terms and all relevant
    relationships (IS_A, PART_OF, REGULATES) into Neo4j.
    """
    if not os.path.exists(go_obo_path):
        logging.warning(f"GO file not found at {go_obo_path}. Skipping ontology import.")
        return False

    logging.info("Parsing GO OBO file with 'pronto' library...")
    try:
        ontology = pronto.Ontology(go_obo_path)
    except Exception as e:
        logging.error(f"Failed to parse OBO file with pronto: {e}")
        return False

    terms_to_import = []
    for term in ontology.terms():
        if term.id.startswith("GO:"):
            terms_to_import.append({"go_id": term.id, "name": term.name, "definition": term.definition or ""})

    logging.info(f"Found {len(terms_to_import)} GO terms to import.")

    # Import nodes in batches
    with driver.session() as session:
        session.run("CREATE INDEX go_term_id_index IF NOT EXISTS FOR (n:GO_Term) ON (n.go_id)").consume()
        node_query = "UNWIND $rows AS row MERGE (g:GO_Term {go_id: row.go_id}) SET g.name = row.name, g.definition = row.definition"
        batch_size = 5000
        logging.info("Importing GO_Term nodes...")
        for i in range(0, len(terms_to_import), batch_size):
            batch = terms_to_import[i:i + batch_size]
            session.run(node_query, rows=batch).consume()
        logging.info("GO_Term node import complete.")

    # This section now parses all relevant relationship types from the go-basic file.
    rels_to_import = []
    # Mapping from OBO relationship IDs to the desired Neo4j relationship type
    rel_map = {
        "is_a": "IS_A",
        "part_of": "PART_OF",
        "regulates": "REGULATES",
        "negatively_regulates": "NEGATIVELY_REGULATES",
        "positively_regulates": "POSITIVELY_REGULATES"
    }

    for term in ontology.terms():
        if term.id.startswith("GO:"):
            for rel_type, superclasses in term.relationships.items():
                # Get the clean relationship name from our map
                rel_name = rel_map.get(rel_type.id)
                if rel_name:
                    for superclass in superclasses:
                        if superclass.id.startswith("GO:"):
                            rels_to_import.append({
                                "subclass_id": term.id, 
                                "superclass_id": superclass.id,
                                "rel_type": rel_name
                            })

    logging.info(f"Found {len(rels_to_import)} total relationships to import.")

    with driver.session() as session:
        # We use APOC here because it allows creating relationships with a dynamic type.
        rel_query = """
        UNWIND $rows AS row
        MATCH (sub:GO_Term {go_id: row.subclass_id})
        MATCH (sup:GO_Term {go_id: row.superclass_id})
        CALL apoc.create.relationship(sub, row.rel_type, {}, sup) YIELD rel
        RETURN count(rel) AS count
        """
        batch_size = 10000
        logging.info("Importing all relationship types...")
        total_rels_created = 0
        for i in range(0, len(rels_to_import), batch_size):
            batch = rels_to_import[i:i + batch_size]
            result_data = session.run(rel_query, rows=batch).data()
            rels_in_batch = sum(record['count'] for record in result_data)
            total_rels_created += rels_in_batch
            logging.info(f"Imported GO relationship batch {i // batch_size + 1}/{ -(-len(rels_to_import) // batch_size)}. Relationships created: {rels_in_batch}")
        logging.info(f"GO relationship import complete. Total relationships created: {total_rels_created}")

    with driver.session() as session:
        result = session.run("MATCH (g:GO_Term) RETURN count(g) as count")
        count = result.single()['count']
        logging.info(f"Verification: Found {count} nodes with the :GO_Term label.")
        return count > 0

def import_gene_go_associations(driver, annotations_path):
    """Reads the VEP output file and creates relationships between Genes and GO_Terms."""
    logging.info(f"Parsing VEP annotation file for GO terms: {annotations_path}")
    try:
        header, all_records, skipped_lines, total_lines = [], [], 0, 0
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if line.startswith('#Uploaded_variation'):
                    header = [h.strip() for h in line.strip().split('\t')]
                    if header and header[0] == '#Uploaded_variation':
                        header[0] = 'Uploaded_variation'
                    continue
                if line.startswith('#') or not line.strip():
                    continue
                fields = [field.strip() for field in line.strip().split('\t')]
                if len(fields) == len(header):
                    all_records.append(dict(zip(header, fields)))
                else:
                    skipped_lines += 1
        logging.info(f"File parsing summary: Total lines read: {total_lines}, Successfully parsed: {len(all_records)}, Skipped lines: {skipped_lines}")

        filtered_records = [r for r in all_records if r.get('SYMBOL') and r.get('GO') and r['SYMBOL'] != '-' and r['GO'] != '-']
        
        association_records = []
        go_id_pattern = re.compile(r"(GO:\d+)")
        for record in filtered_records:
            gene_symbol = record['SYMBOL']
            go_ids_found = go_id_pattern.findall(record['GO'])
            for go_id in go_ids_found:
                association_records.append({'gene_symbol': gene_symbol, 'go_id': go_id})
        
        unique_records = [dict(t) for t in {tuple(d.items()) for d in association_records}]
        logging.info(f"Found {len(unique_records)} unique gene-GO term associations to import.")

        query = """
        UNWIND $rows AS row
        MATCH (g:Gene {symbol: row.gene_symbol})
        MATCH (go:GO_Term {go_id: row.go_id})
        MERGE (g)-[:HAS_GO_TERM]->(go)
        """
        batch_size = 50000
        with driver.session() as session:
            logging.info("Creating relationships between genes and GO terms...")
            total_rels_created = 0
            for i in range(0, len(unique_records), batch_size):
                batch = unique_records[i:i + batch_size]
                result = session.run(query, rows=batch)
                summary = result.consume()
                rels_created = summary.counters.relationships_created
                total_rels_created += rels_created
                logging.info(f"Imported association batch {i // batch_size + 1}. Relationships created: {rels_created}")
            logging.info(f"Gene-GO Term relationship import complete. Total relationships created: {total_rels_created}")
    except Exception as e:
        logging.error(f"Error processing GO association file: {e}")

# --- Main execution block ---
def main_go_pipeline():
    """Orchestrates the download and import of GO data."""
    try:
        logging.info("--- Starting Gene Ontology (GO) Import Pipeline ---")
        time.sleep(10)
        go_obo_url = "https://current.geneontology.org/ontology/go-basic.obo"
        go_obo_path = download_file(go_obo_url, DOWNLOAD_DIR, "go-basic.obo")
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            logging.info("Clearing any previous GO data...")
            session.run("MATCH (n:GO_Term) DETACH DELETE n").consume()

        # Import the ontology structure first
        ontology_imported = import_go_ontology_manual(driver, go_obo_path)
        
        # Then import the gene-GO term links from your VEP file
        if ontology_imported:
            import_gene_go_associations(driver, ANNOTATED_TSV_PATH)
        else:
            logging.error("GO Ontology import failed. Aborting gene-GO term association import.")

        driver.close()
        logging.info("--- GO Import Pipeline Finished Successfully! ---")
    except Exception as e:
        logging.error(f"An error occurred in the GO pipeline: {e}")

if __name__ == "__main__":
    main_go_pipeline()
