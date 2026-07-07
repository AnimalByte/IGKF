import os
import logging
import urllib.request
import pandas as pd
import time
import shutil
from config import get_neo4j_driver, DOWNLOAD_DIR

# --- Data URLs and Filenames ---
ENCODE_CCRE_URL = "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"

# --- Data Download Function ---
def download_file(url, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        logging.info(f"Downloading {filename} from {url}...")
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response, open(filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        logging.info("Download complete.")
    else:
        logging.info(f"{filename} already exists. Skipping download.")
    return filepath

def import_encode_data(driver, bed_path):
    if not os.path.exists(bed_path):
        logging.error(f"ENCODE data file not found at {bed_path}.")
        return False
    logging.info("Parsing ENCODE cCRE data...")
    try:
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2, 3, 4, 5])
        df.columns = ['chrom', 'start', 'end', 'accession', 'evidence_accession', 'label']
        records = df.to_dict('records')
        logging.info(f"Found {len(records)} candidate cCREs to import.")
        query = """
        UNWIND $rows AS row
        CREATE (re:RegulatoryElement {
            accession: row.accession,
            evidence_accession: row.evidence_accession,
            label: row.label,
            location: row.chrom + ':' + row.start + '-' + row.end,
            chrom: row.chrom,
            start: toInteger(row.start),
            end: toInteger(row.end)
        })
        """
        with driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:RegulatoryElement) ON (n.accession)").consume()
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:RegulatoryElement) ON (n.chrom)").consume()
            session.run("CREATE RANGE INDEX IF NOT EXISTS FOR (n:RegulatoryElement) ON (n.start, n.end)").consume()
            logging.info("Importing ENCODE cCREs into Neo4j...")
            batch_size = 50000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                summary = session.run(query, rows=batch).consume()
                logging.info(f"Imported cCRE chunk {i//batch_size + 1}. Nodes created: {summary.counters.nodes_created}")
        logging.info("ENCODE cCRE data import complete.")
        return True
    except Exception as e:
        logging.error(f"An error occurred during ENCODE import: {e}")
        return False

def link_variants_to_regulatory_elements(driver):
    logging.info("Linking GWAS variants to overlapping ENCODE regulatory elements...")
    try:
        with driver.session() as session:
            link_query = """
            CALL apoc.periodic.iterate(
                "MATCH (m:Mutation) WHERE m.chrom IS NOT NULL AND m.pos IS NOT NULL RETURN m",
                "WITH m
                 MATCH (re:RegulatoryElement {chrom: m.chrom})
                 WHERE toInteger(m.pos) >= re.start AND toInteger(m.pos) <= re.end
                 MERGE (m)-[:IN_REGULATORY_ELEMENT]->(re)",
                {batchSize: 5000, parallel: true, retries: 3}
            ) YIELD batches, total, timeTaken
            RETURN batches, total, timeTaken
            """
            logging.info("Executing optimized linking query...")
            result = session.run(link_query).single()
            if result:
                logging.info(f"Finished linking variants. Processed {result['total']} mutations in {result['batches']} batches. Time: {result['timeTaken']}s.")
            else:
                logging.error("Linking query did not return expected results.")
    except Exception as e:
        logging.error(f"An error occurred while linking variants: {e}", exc_info=True)
        logging.warning("APOC-based linking failed.")

def main_encode_pipeline():
    try:
        logging.info("--- Starting ENCODE cCRE Data Import Pipeline ---")
        time.sleep(5)
        
        ccre_bed_path = download_file(ENCODE_CCRE_URL, DOWNLOAD_DIR, "GRCh38-cCREs.bed")
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            logging.info("Clearing any previous ENCODE data...")
            while True:
                result = session.run("MATCH (re:RegulatoryElement) WITH re LIMIT 100000 DETACH DELETE re RETURN count(re)").consume()
                if result.counters.nodes_deleted == 0:
                    break
                logging.info(f"Deleted a batch of old ENCODE nodes.")
            logging.info("Finished clearing old ENCODE data.")
            
        if import_encode_data(driver, ccre_bed_path):
            link_variants_to_regulatory_elements(driver)
        
        driver.close()
        logging.info("--- ENCODE Data Import Pipeline Finished Successfully! ---")
    except Exception as e:
        logging.error(f"An error occurred in the ENCODE pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main_encode_pipeline()
