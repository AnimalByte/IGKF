import os
import time
import logging
from Bio import Entrez, Medline
from datetime import datetime, timedelta
from config import get_neo4j_driver, ABSTRACTS_DIR as OUTPUT_DIR

# --- FIX: Dynamic API Rate Limiting & Configuration ---
# Read email and API key from environment variables
Entrez.email = os.getenv("NCBI_EMAIL", "your_email@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY") 

if Entrez.email == "your_email@example.com":
    logging.warning("NCBI_EMAIL environment variable not set. Using a default email. Please set your own email for NCBI.")

if Entrez.api_key:
    logging.info("NCBI API key found. Using higher rate limits.")
    MAX_PAPERS_PER_GENE = 10
    PAPERS_PER_REQUEST = 10
    API_DELAY = 0.1  # 10 requests per second
else:
    logging.warning("NCBI_API_KEY not set. Using lower rate limits.")
    MAX_PAPERS_PER_GENE = 9
    PAPERS_PER_REQUEST = 3
    API_DELAY = 0.34 # ~3 requests per second
# --- END FIX ---

SEARCH_YEARS_PRIMARY = 7
SEARCH_YEARS_FALLBACK = 15

def get_genes_from_neo4j(driver):
    """
    Queries the Neo4j database to get a list of gene symbols that are
    directly associated with a Mutation from the VCF file.
    """
    logging.info("Fetching gene symbols directly linked to mutations from Neo4j...")
    genes = []
    try:
        with driver.session() as session:
            query = """
            MATCH (m:Mutation)-[:AFFECTS]->(g:Gene)
            RETURN DISTINCT g.symbol AS symbol
            """
            result = session.run(query)
            genes = [record["symbol"] for record in result]
        logging.info(f"Found {len(genes)} unique gene symbols linked to mutations to process.")
        return genes
    except Exception as e:
        logging.error(f"Failed to query genes from Neo4j: {e}")
        return []

def search_pubmed(gene_symbol, years):
    """Performs a PubMed search for a given gene and time window."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    start_date_str = start_date.strftime("%Y/%m/%d")
    end_date_str = end_date.strftime("%Y/%m/%d")
    
    search_term = (
        f'({gene_symbol}[Gene Name] OR "{gene_symbol}"[All Fields]) AND '
        f'"human"[Organism] AND hasabstract[Filter] AND '
        f'("{start_date_str}"[Date - Publication] : "{end_date_str}"[Date - Publication])'
    )
    
    logging.info(f"Searching PubMed for '{gene_symbol}' within the last {years} years...")
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=MAX_PAPERS_PER_GENE)
    record = Entrez.read(handle)
    handle.close()
    time.sleep(API_DELAY) # Adhere to rate limit after each search
    return record["IdList"]

def fetch_pubmed_abstracts(gene_symbol):
    """
    Searches PubMed for a given gene symbol and downloads abstracts in chunks,
    respecting the API rate limits.
    """
    try:
        pmids = search_pubmed(gene_symbol, SEARCH_YEARS_PRIMARY)

        if not pmids:
            logging.warning(f"No results for '{gene_symbol}' in last {SEARCH_YEARS_PRIMARY} years. Expanding search.")
            pmids = search_pubmed(gene_symbol, SEARCH_YEARS_FALLBACK)

        if not pmids:
            logging.info(f"No results found for '{gene_symbol}'. Skipping.")
            return

        logging.info(f"Found {len(pmids)} PMIDs for '{gene_symbol}'.")

        saved_count = 0
        for i in range(0, len(pmids), PAPERS_PER_REQUEST):
            chunk = pmids[i:i + PAPERS_PER_REQUEST]
            logging.info(f"Fetching chunk {i//PAPERS_PER_REQUEST + 1} for {gene_symbol}...")
            
            handle = Entrez.efetch(db="pubmed", id=chunk, rettype="medline", retmode="text")
            records = Medline.parse(handle)
            
            gene_dir = os.path.join(OUTPUT_DIR, gene_symbol)
            if not os.path.exists(gene_dir):
                os.makedirs(gene_dir)

            for rec in records:
                abstract = rec.get("AB")
                pmid = rec.get("PMID")
                if abstract:
                    filepath = os.path.join(gene_dir, f"{pmid}.txt")
                    if os.path.exists(filepath):
                        logging.info(f"Skipping existing abstract for PMID {pmid}.")
                        continue
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(abstract)
                    saved_count += 1
            
            time.sleep(API_DELAY) # Adhere to rate limit after each fetch

        logging.info(f"Successfully saved {saved_count} abstracts for {gene_symbol}.")
        
    except Exception as e:
        logging.error(f"An error occurred while fetching data for {gene_symbol}: {e}")

def main():
    """Main function to run the PubMed data fetching pipeline."""
    try:
        driver = get_neo4j_driver()
        
        genes = get_genes_from_neo4j(driver)
        
        if not genes:
            logging.warning("No genes found in Neo4j. Cannot fetch PubMed data.")
            return

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        logging.info(f"Starting PubMed abstract download for {len(genes)} genes...")
        for gene in genes:
            fetch_pubmed_abstracts(gene)
            
        driver.close()
        logging.info("PubMed abstract fetching complete!")

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main()
