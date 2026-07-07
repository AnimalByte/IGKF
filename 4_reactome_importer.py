import os
import logging
import pandas as pd
import time
import numpy as np
from config import get_neo4j_driver, download_file, ANNOTATED_TSV_PATH, DOWNLOAD_DIR

# --- Combined Import and Enrichment Functions ---

def clean_uniprot_id(uniprot_id):
    """Removes version suffixes (e.g., .1) from UniProt IDs."""
    if isinstance(uniprot_id, str):
        return uniprot_id.split('.')[0]
    return uniprot_id

def enrich_gene_nodes(driver, tsv_path):
    """
    Reads the VEP TSV file and adds additional identifiers (UniProt IDs)
    to the existing :Gene nodes in the Neo4j database. This version uses a
    robust parser to handle the VEP output format correctly.
    """
    if not os.path.exists(tsv_path):
        print(f"ERROR: TSV file not found at {tsv_path}. Cannot enrich gene nodes.")
        return False

    print(f"Step 1: Reading data from {tsv_path} to enrich Gene nodes...")
    try:
        header = []
        all_records = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
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
        
        if not all_records:
            print("WARNING: No valid data records found in VEP output file.")
            return False

        df_full = pd.DataFrame(all_records)
        df = df_full[['SYMBOL', 'SWISSPROT', 'TREMBL']].copy()
        df.replace('-', np.nan, inplace=True)
        df.dropna(subset=['SYMBOL'], inplace=True)

        print("Cleaning UniProt ID version suffixes...")
        df['SWISSPROT'] = df['SWISSPROT'].apply(clean_uniprot_id)
        df['TREMBL'] = df['TREMBL'].apply(clean_uniprot_id)
        
        print("Aggregating unique identifiers for each gene symbol...")
        gene_info_df = df.groupby('SYMBOL').first().reset_index()
        gene_info_df.fillna('-', inplace=True)
        
        records = gene_info_df.to_dict('records')
        print(f"Found {len(records)} unique genes to enrich.")

        query = """
        UNWIND $rows AS row
        MATCH (g:Gene {symbol: row.SYMBOL})
        SET g.swissprot_id = CASE WHEN row.SWISSPROT <> '-' THEN row.SWISSPROT ELSE null END,
            g.trembl_id = CASE WHEN row.TREMBL <> '-' THEN row.TREMBL ELSE null END
        """
        
        with driver.session() as session:
            print("Updating Gene nodes with UniProt properties...")
            result = session.run(query, rows=records)
            summary = result.consume()
            print(f"Processed gene enrichment batch. Properties set: {summary.counters.properties_set}")
        
        print("Gene node enrichment complete.")
        return True
        
    except Exception as e:
        print(f"An error occurred during gene enrichment: {e}")
        return False

def import_reactome_data(driver, associations_path, pathway_path):
    """Imports Reactome pathways and creates relationships to the now-enriched Gene nodes."""
    print("Step 2: Importing Reactome data...")
    try:
        # Part A: Create Pathway nodes
        pathway_df = pd.read_csv(pathway_path, sep='\t', header=None, usecols=[0, 1])
        pathway_df.columns = ['pathway_id', 'pathway_name']
        pathway_records = pathway_df.to_dict('records')
        
        with driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Pathway) ON (n.id)").consume()
            pathway_query = "UNWIND $rows AS row MERGE (p:Pathway {id: row.pathway_id}) SET p.name = row.pathway_name"
            session.run(pathway_query, rows=pathway_records).consume()
            print(f"Successfully created or merged {len(pathway_records)} Pathway nodes.")

        # Part B: Build an in-memory map of UniProt IDs to Gene Symbols from the graph
        print("Building UniProt ID to Gene Symbol map from Neo4j...")
        uniprot_to_gene_map = {}
        with driver.session() as session:
            result = session.run("MATCH (g:Gene) WHERE g.swissprot_id IS NOT NULL OR g.trembl_id IS NOT NULL RETURN g.symbol, g.swissprot_id, g.trembl_id")
            for record in result:
                if record["g.swissprot_id"]:
                    uniprot_to_gene_map[record["g.swissprot_id"]] = record["g.symbol"]
                if record["g.trembl_id"]:
                    uniprot_to_gene_map[record["g.trembl_id"]] = record["g.symbol"]
        print(f"Built map with {len(uniprot_to_gene_map)} UniProt ID entries.")

        # Part C: Create relationships using the in-memory map
        assoc_df = pd.read_csv(associations_path, sep='\t', header=None, usecols=[0, 1])
        assoc_df.columns = ['uniprot_id', 'pathway_id']
        
        # --- FIX: Filter out EC numbers and other non-UniProt IDs ---
        # A valid UniProt ID does not contain periods.
        assoc_df = assoc_df[~assoc_df['uniprot_id'].astype(str).str.contains(r'\.')]
        print(f"Filtered out non-UniProt identifiers. Keeping {len(assoc_df)} records.")
        # --- END FIX ---
        
        relationships_to_create = []
        for index, row in assoc_df.iterrows():
            uniprot_id = row['uniprot_id']
            if uniprot_id in uniprot_to_gene_map:
                relationships_to_create.append({
                    "gene_symbol": uniprot_to_gene_map[uniprot_id],
                    "pathway_id": row['pathway_id']
                })
        
        print(f"Found {len(relationships_to_create)} matching associations to import.")
        
        rel_query = """
        UNWIND $rows AS row
        MATCH (g:Gene {symbol: row.gene_symbol})
        MATCH (p:Pathway {id: row.pathway_id})
        MERGE (g)-[:PARTICIPATES_IN]->(p)
        """
        
        batch_size = 50000
        with driver.session() as session:
            print("Creating relationships between genes and pathways...")
            total_rels_created = 0
            for i in range(0, len(relationships_to_create), batch_size):
                batch = relationships_to_create[i:i + batch_size]
                result = session.run(rel_query, rows=batch)
                summary = result.consume()
                rels_created_this_batch = summary.counters.relationships_created
                total_rels_created += rels_created_this_batch
                print(f"Imported relationship batch {i // batch_size + 1}. Relationships created: {rels_created_this_batch}")
            print(f"Gene-pathway relationship import complete. Total relationships created: {total_rels_created}")

    except Exception as e:
        print(f"An error occurred during Reactome import: {e}")

# --- Main execution block ---
def main_pipeline():
    """Orchestrates the download and import of Reactome data."""
    try:
        print("--- Starting Consolidated Reactome Import Pipeline ---")
        
        time.sleep(5)

        uniprot_to_pathway_url = "https://reactome.org/download/current/UniProt2Reactome.txt"
        pathway_names_url = "https://reactome.org/download/current/ReactomePathways.txt"

        associations_path = download_file(uniprot_to_pathway_url, DOWNLOAD_DIR, "UniProt2Reactome.txt")
        pathway_names_path = download_file(pathway_names_url, DOWNLOAD_DIR, "ReactomePathways.txt")

        driver = get_neo4j_driver()
        
        enrichment_successful = enrich_gene_nodes(driver, ANNOTATED_TSV_PATH)
        
        if enrichment_successful:
            with driver.session() as session:
                print("Clearing any previous Reactome pathway data...")
                session.run("MATCH (n:Pathway) DETACH DELETE n").consume()
            import_reactome_data(driver, associations_path, pathway_names_path)
        else:
            print("ERROR: Gene enrichment failed. Aborting Reactome import.")

        driver.close()
        print(f"--- Consolidated Reactome Import Pipeline Finished Successfully! ---")

    except Exception as e:
        print(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main_pipeline()
