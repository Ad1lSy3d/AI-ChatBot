import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define strict local paths
RAW_DATA_DIR = Path("data/raw")
VECTOR_STORE_DIR = Path("vector_store")

def run_ingestion():
    print("Starting Bharti AXA Life Ingestion Pipeline...")
    
    # 1. Gather all PDFs
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{RAW_DATA_DIR}'. Please drop your files there first.")
        return

    print(f"📁 Found {len(pdf_files)} PDFs to process.")
    
    all_chunks = []
    
    # 2. Extract and Chunk each file with domain-aware parameters
    for pdf_path in pdf_files:
        filename = pdf_path.name.lower()
        print(f"📄 Processing: {pdf_path.name}...")
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Context-Aware Parameter Switching based on document type
            if "brochure" in filename:
                # Brochures: Shorter chunks for visual, scannable layouts
                chunk_size = 1000
                chunk_overlap = 150
                print("   🏷️ Document Type: Brochure -> Applying tight chunking parameters.")
            else:
                # Policy text bonds/legal files: Longer blocks to preserve clause strings
                chunk_size = 1800
                chunk_overlap = 300
                print("   🏷️ Document Type: Legal/Policy -> Applying protective clause overlap.")
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            
            file_chunks = splitter.split_documents(documents)
            all_chunks.extend(file_chunks)
            print(f"Generated {len(file_chunks)} text chunks.")
            
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {str(e)}")
            continue

    if not all_chunks:
        print("Ingestion halted: No text chunks were successfully extracted.")
        return

    print(f"Total chunks compiled across knowledge base: {len(all_chunks)}")

    # 3. Initialize Local Embedding Engine
    print(" Initializing Local SentenceTransformers (all-MiniLM-L6-v2)...")
    # This downloads the model cleanly on first run, processing purely on local CPU/RAM
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Generate Vectors & Serialize into local FAISS Index
    print("⚡ Generating embeddings and building FAISS index (this may take a few moments)...")
    db = FAISS.from_documents(all_chunks, embeddings)
    
    # Ensure directory exists before writing
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    db.save_local(str(VECTOR_STORE_DIR))
    
    print(f"🎉 Success! FAISS index successfully generated and saved to '{VECTOR_STORE_DIR}/'")

if __name__ == "__main__":
    run_ingestion()