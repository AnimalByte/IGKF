import os
import logging
import gzip
import time
from config import get_neo4j_driver, download_file, DOWNLOAD_DIR

# --- ClinVar Import Function ---

def parse_info_field(info_str):
    """Parses the INFO column of a VCF file into a dictionary."""
    info_dict = {}
    for item in info_str.split(';'):
        if '=' in item:
            key, value = item.split('=', 1)
            info_dict[key] = value
    return info_dict

def import_clinvar_data(driver, vcf_path):
    """
    Parses the ClinVar VCF file and enriches existing :Mutation nodes with
    clinical significance by matching on rsID.
    """
    if not os.path.exists(vcf_path):
        logging.error(f"ClinVar VCF file not found at {vcf_path}. Cannot proceed.")
        return

    # --- Step 1: Get all mutation rsIDs from our graph ---
    graph_mutation_ids = set()
    with driver.session() as session:
        logging.info("Fetching all mutation IDs from the Neo4j graph...")
        result = session.run("MATCH (m:Mutation) RETURN m.id AS rsid")
        for record in result:
            graph_mutation_ids.add(record["rsid"])
    
    if not graph_mutation_ids:
        logging.error("No Mutation nodes found in the graph. Please run the 'neo4j_base_importer.py' script first.")
        return
    logging.info(f"Found {len(graph_mutation_ids)} unique mutation IDs in the graph to match against.")
    logging.info(f"Sample of graph IDs: {list(graph_mutation_ids)[:5]}")


    # --- Step 2: Parse the ClinVar VCF and find matching records ---
    logging.info("Parsing ClinVar VCF file to find matching variants...")
    
    records_to_import = []
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            info_dict = parse_info_field(fields[7])
            
            rs_id_from_info = info_dict.get('RS')
            
            if rs_id_from_info and 'CLNSIG' in info_dict:
                # Add the 'rs' prefix to match the format in our graph
                full_rs_id = "rs" + rs_id_from_info
                
                # **DEBUGGING & EFFICIENCY FIX**: Only process if the rsID exists in our graph
                if full_rs_id in graph_mutation_ids:
                    records_to_import.append({
                        "rs_id": full_rs_id,
                        "clnsig": info_dict.get('CLNSIG', 'Unknown'),
                        "clndn": info_dict.get('CLNDN', 'Unknown').replace('_', ' ')
                    })

    logging.info(f"Finished parsing. Found {len(records_to_import)} matching records in ClinVar to import.")
    if records_to_import:
        logging.info(f"Sample of matching ClinVar record to be loaded: {records_to_import[0]}")

    # --- Step 3: Enrich the graph with the matching ClinVar data ---
    if not records_to_import:
        logging.warning("No matching variants found in ClinVar. No data will be loaded.")
        return

    query = """
    UNWIND $rows AS row
    MATCH (m:Mutation {id: row.rs_id})
    SET m.clinical_significance = row.clnsig
    WITH m, row
    WHERE row.clndn IS NOT NULL AND row.clndn <> 'Unknown' AND row.clndn <> 'not_provided'
    UNWIND split(row.clndn, '|') AS disease_name_part
    UNWIND split(disease_name_part, ',') AS disease_name
    MERGE (d:Disease {name: disease_name})
    MERGE (m)-[:ASSOCIATED_DISEASE]->(d)
    """
    
    batch_size = 25000
    with driver.session() as session:
        session.run("CREATE INDEX disease_name_index IF NOT EXISTS FOR (n:Disease) ON (n.name)").consume()
        
        logging.info("Enriching graph with matched ClinVar data...")
        total_props_set = 0
        total_rels_created = 0

        for i in range(0, len(records_to_import), batch_size):
            batch = records_to_import[i:i + batch_size]
            result = session.run(query, rows=batch)
            summary = result.consume()
            props_set = summary.counters.properties_set
            rels_created = summary.counters.relationships_created
            total_props_set += props_set
            total_rels_created += rels_created
            logging.info(f"Imported ClinVar batch {i // batch_size + 1}. Properties set: {props_set}, Relationships created: {rels_created}")
            
        logging.info(f"ClinVar data import complete. Total properties set: {total_props_set}, Total relationships created: {total_rels_created}.")

# --- Main execution block ---
def main_clinvar_pipeline():
    """Orchestrates the download and import of ClinVar data."""
    try:
        logging.info("--- Starting ClinVar Data Import Pipeline ---")
        time.sleep(5)
        clinvar_vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
        vcf_path = download_file(clinvar_vcf_url, DOWNLOAD_DIR, "clinvar.vcf.gz")
        driver = get_neo4j_driver()
        import_clinvar_data(driver, vcf_path)
        driver.close()
        logging.info("--- ClinVar Data Import Pipeline Finished Successfully! ---")
    except Exception as e:
        logging.error(f"An error occurred in the ClinVar pipeline: {e}")

if __name__ == "__main__":
    main_clinvar_pipeline()
