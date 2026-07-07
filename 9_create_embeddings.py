import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from chromadb.config import Settings
from config import (
    ABSTRACTS_DIR, CHROMA_DB_PATH as DB_PATH,
    CHROMA_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL_NAME as MODEL_NAME,
)

# --- Main Script ---
def create_and_store_embeddings():
    """
    Scans the abstracts, generates embeddings using a biomedical-specific
    transformer model, and stores them in a local ChromaDB vector database.
    """
    logging.info("--- Starting Embedding Creation and Storage ---")

    # --- Step 1: Initialize Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    logging.info(f"Loading sentence transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logging.info("Model loaded successfully.")

    # --- Step 2: Initialize Vector Database ---
    # --- FIX: Use the modern, settings-based client for consistency ---
    client = chromadb.Client(Settings(persist_directory=DB_PATH, is_persistent=True))
    # --- END FIX ---
    
    logging.info(f"Initializing ChromaDB collection: '{COLLECTION_NAME}'")
    # Use get_or_create_collection to ensure it's created if it doesn't exist
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing_ids_results = collection.get(include=[]) 
    processed_ids = set(existing_ids_results['ids'])
    logging.info(f"Found {len(processed_ids)} existing documents in the database. They will be skipped.")

    # --- Step 3: Process Abstracts and Create Embeddings ---
    logging.info(f"Scanning for abstracts in: '{ABSTRACTS_DIR}'")
    
    documents_batch = []
    metadatas_batch = []
    ids_batch = []
    batch_size = 50
    
    total_files_scanned = 0
    for gene_symbol in os.listdir(ABSTRACTS_DIR):
        gene_dir = os.path.join(ABSTRACTS_DIR, gene_symbol)
        if os.path.isdir(gene_dir):
            for filename in os.listdir(gene_dir):
                if filename.endswith(".txt"):
                    total_files_scanned += 1
                    pmid = os.path.splitext(filename)[0]

                    if pmid in processed_ids:
                        continue

                    filepath = os.path.join(gene_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        abstract_text = f.read()

                    documents_batch.append(abstract_text)
                    metadatas_batch.append({"gene": gene_symbol, "pmid": pmid})
                    ids_batch.append(pmid)
                    processed_ids.add(pmid)
                    
                    if len(documents_batch) >= batch_size:
                        logging.info(f"Processing batch of {len(documents_batch)} new abstracts...")
                        embeddings = model.encode(documents_batch, show_progress_bar=True)
                        collection.add(
                            embeddings=embeddings.tolist(),
                            documents=documents_batch,
                            metadatas=metadatas_batch,
                            ids=ids_batch
                        )
                        logging.info(f"Successfully stored batch. Total unique documents in DB: {collection.count()}")
                        documents_batch, metadatas_batch, ids_batch = [], [], []

    if documents_batch:
        logging.info(f"Processing final batch of {len(documents_batch)} new abstracts...")
        embeddings = model.encode(documents_batch, show_progress_bar=True)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents_batch,
            metadatas=metadatas_batch,
            ids=ids_batch
        )

    logging.info("--- Embedding Creation and Storage Complete! ---")
    logging.info(f"Total files scanned: {total_files_scanned}")
    logging.info(f"Total unique documents in database: {collection.count()}")
    logging.info(f"Vector database is stored at: '{DB_PATH}'")


if __name__ == "__main__":
    create_and_store_embeddings()

