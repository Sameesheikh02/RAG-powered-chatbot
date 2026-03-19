import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from vector import retriever # Ensure this is optimized to only load the DB once

# --- 1. Page Configuration ---
st.set_page_config(page_title="Pizza RAG Chat", page_icon="🍕")

# --- 2. Cache Resources (Efficiency Boost) ---
# This prevents reloading the model/chain every time the user clicks a button
@st.cache_resource
def get_llm_chain(model_name):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Pizza Restaurant Assistant.
        Use the provided review context to answer questions. 
        - If the context is relevant, summarize the feedback.
        - If no context is found or it's irrelevant, use your general knowledge but stay in character.
        - Always be enthusiastic about pizza!
        
        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    llm = OllamaLLM(model=model_name, temperature=0.3)
    return prompt | llm

# --- 3. Sidebar & State ---
with st.sidebar:
    st.title("Settings")
    model_choice = st.selectbox("Ollama Model", ["llama3.2"], index=0)
    show_docs = st.toggle("Show retrieved sources", value=True)
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_docs = []
        st.rerun()

# Initialize History & Chain
msgs = StreamlitChatMessageHistory(key="chat_history")
chain = get_llm_chain(model_choice)

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# --- 4. UI Display ---
st.title("🍕 Pizza Review Assistant")

# Render existing messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# --- 5. Chat Logic ---
if question := st.chat_input("How's the pepperoni pizza?"):
    # 1. Display User Message
    st.chat_message("human").write(question)
    
    # 2. RAG Logic: Greeting Check (Efficiency: avoid vector search for 'hi')
    greetings = {"hi", "hello", "hey", "hola", "yo"}
    is_greeting = question.lower().strip() in greetings
    
    retrieved_docs = []
    if not is_greeting:
        with st.spinner("Checking reviews..."):
            # Efficiency: invoke returns docs, we store them once
            retrieved_docs = retriever.invoke(question)
            st.session_state.last_docs = retrieved_docs
            context_text = "\n\n".join([d.page_content for d in retrieved_docs])
    else:
        context_text = "The user is just saying hello. Be friendly!"

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Build inputs
        inputs = {
            "context": context_text,
            "history": msgs.messages,
            "question": question
        }

        # Stream the response
        for chunk in chain.stream(inputs):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Store in history
        msgs.add_user_message(question)
        msgs.add_ai_message(full_response)

# --- 6. Source Display ---
if show_docs and st.session_state.last_docs:
    st.divider()
    with st.expander("🔍 Evidence from Reviews", expanded=False):
        for i, doc in enumerate(st.session_state.last_docs):
            rating = doc.metadata.get('rating', 'N/A')
            st.markdown(f"**Review {i+1}** (⭐ {rating}/5)")
            st.info(doc.page_content)
