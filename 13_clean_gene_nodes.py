import logging
import re
from config import get_neo4j_driver

def clean_gene_nodes(driver):
    """
    Finds and deletes :Gene nodes that have invalid symbols.
    """
    logging.info("--- Starting Gene Node Cleaning ---")
    
    # This query finds all nodes labeled :Gene
    get_genes_query = "MATCH (g:Gene) RETURN g.symbol AS symbol"
    
    # This query will delete a specific bad node and its relationships
    delete_gene_query = "MATCH (g:Gene {symbol: $symbol}) DETACH DELETE g"

    # Regex for a valid gene symbol (starts with a letter, no spaces, etc.)
    # This is a general pattern; specific gene naming conventions can be complex.
    valid_symbol_pattern = re.compile(r'^[A-Z][A-Z0-9-]*[A-Z0-9]$', re.IGNORECASE)
    
    deleted_count = 0
    
    with driver.session() as session:
        logging.info("Fetching all gene symbols from the database...")
        results = session.run(get_genes_query)
        all_symbols = [record["symbol"] for record in results]
        
        logging.info(f"Found {len(all_symbols)} total gene symbols. Checking for invalid entries...")
        
        for symbol in all_symbols:
            # Check for invalid symbols
            is_invalid = False
            if not isinstance(symbol, str):
                is_invalid = True
            # Rule 1: Must not contain spaces (like '-PIK3R3 readthrough')
            elif ' ' in symbol:
                is_invalid = True
            # Rule 2: Must not be an EC number (like '1.1.1.271')
            elif '.' in symbol:
                is_invalid = True
            # Rule 3: Must match a general pattern for symbols (not just numbers)
            elif not valid_symbol_pattern.match(symbol):
                is_invalid = True

            if is_invalid:
                logging.warning(f"Found invalid gene symbol: '{symbol}'. Deleting node and relationships.")
                session.run(delete_gene_query, symbol=symbol)
                deleted_count += 1
                
    logging.info(f"Cleaning complete. Deleted {deleted_count} invalid gene nodes.")

def main():
    """Main function to run the cleaning pipeline."""
    try:
        driver = get_neo4j_driver()
        clean_gene_nodes(driver)
        driver.close()
        logging.info("--- Gene Node Cleaning Finished Successfully! ---")
    except Exception as e:
        logging.error(f"An error occurred in the cleaning pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()
