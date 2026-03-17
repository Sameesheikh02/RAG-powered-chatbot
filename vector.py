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
    # 1. Setup Embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # 2. Initialize or Load Chroma DB
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    # 3. Only ingest data if the DB is empty or doesn't exist
    if not os.path.exists(DB_LOCATION) or vector_store._collection.count() == 0:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            documents = []
            for i, row in df.iterrows():
                # Combine Title and Review for better search context
                content = f"Review Title: {row['Title']}\nContent: {row['Review']}"
                doc = Document(
                    page_content=content,
                    metadata={"rating": row.get("Rating", "N/A"), "date": row.get("Date", "N/A")},
                    id=str(i)
                )
                documents.append(doc)
            
            vector_store.add_documents(documents)
            print("--- Database created and documents added ---")
        else:
            st.error(f"Critical Error: {CSV_FILE} not found!")

    # 4. Return as retriever
    return vector_store.as_retriever(search_kwargs={"k": 5})

# Export retriever instance
retriever = get_retriever()