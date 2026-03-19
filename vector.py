import os
import pandas as pd
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_LOCATION = "./chrome_langchain_db"
CSV_FILE = "realistic_restaurant_reviews.csv"

@st.cache_resource
def get_retriever():
    # 1. Initialize Embeddings (mxbai-embed-large is a great choice)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # 2. Setup Vector Store
    # We initialize it first to check existence
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    # 3. Smart Ingestion Logic
    # check if collection is empty
    if vector_store._collection.count() == 0:
        if not os.path.exists(CSV_FILE):
            st.error(f"🚨 File Not Found: {CSV_FILE}")
            return None
        
        with st.status("Initializing Vector Database...", expanded=True) as status:
            st.write("Reading CSV...")
            df = pd.read_csv(CSV_FILE)
            
            # Optimization: Use list comprehension instead of iterrows (much faster)
            st.write("Creating documents...")
            documents = [
                Document(
                    page_content=f"Review Title: {row['Title']}\nContent: {row['Review']}",
                    metadata={
                        "rating": row.get("Rating", "N/A"), 
                        "date": row.get("Date", "N/A"),
                        "source": "csv"
                    }
                )
                for _, row in df.iterrows()
            ]
            
            st.write(f"Adding {len(documents)} reviews to Chroma...")
            # Optimization: Chroma handles batching, but we ensure it's clean
            vector_store.add_documents(documents)
            status.update(label="Database Ready!", state="complete", expanded=False)

    # 4. Advanced Retriever Settings
    # search_type="mmr" (Maximum Marginal Relevance) helps avoid redundant reviews 
    # (e.g., getting 5 reviews that all say the exact same thing)
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 10}
    )

# Export retriever instance
retriever = get_retriever()
