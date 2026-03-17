import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from vector import retriever

# --- 1. Page Configuration ---
st.set_page_config(page_title="Pizza RAG Chat", page_icon="🍕", layout="centered")

st.title("🍕 Pizza Review Assistant")
st.markdown("Ask me anything about our pizza reviews!")

# Initialize session state for docs so they persist across reruns (toggles)
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# --- 2. Sidebar Settings ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Ollama Model", ["llama3.2"], index=0)
    show_docs = st.toggle("Show retrieved sources", value=True)
    
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.last_docs = [] # Clear docs too
        st.rerun()
    
    st.divider()
    st.caption("Status: Local Ollama Connected")

# --- 3. Memory & Chat History ---
msgs = StreamlitChatMessageHistory(key="chat_history")

# --- 4. Define the AI Brain (LangChain) ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for a pizza restaurant. 
    Use the provided reviews context to answer the user's question. 
    If the context is empty or not relevant, just be polite.
    
    Context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

llm = OllamaLLM(model=model_choice, temperature=0.3)
chain = prompt | llm

# --- 5. Display Existing Messages ---
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# --- 6. The "Gatekeeper" Logic & Chat Input ---
if question := st.chat_input("How's the pepperoni pizza?"):
    
    st.chat_message("human").write(question)
    
    greetings = ["hi", "hello", "hey", "hola", "yo", "good morning"]
    is_greeting = question.lower().strip() in greetings

    context_text = ""
    st.session_state.last_docs = [] # Reset for new question

    if not is_greeting:
        # Use st.spinner instead of st.status so it disappears when done
        with st.spinner("Searching through reviews..."):
            retrieved_docs = retriever.invoke(question)
            st.session_state.last_docs = retrieved_docs 
            context_text = "\n\n".join([d.page_content for d in retrieved_docs])
    else:
        context_text = "N/A (Greeting detected)"

    # The spinner is now gone. Now we start the Assistant response.
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        inputs = {
            "context": context_text,
            "history": msgs.messages,
            "question": question
        }

        # The user will only see the typing effect now
        for chunk in chain.stream(inputs):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Save history after the full response is generated
        msgs.add_user_message(question)
        msgs.add_ai_message(full_response)

# --- 7. Display Sources (Outside the input block) ---
# This ensures they show up even when you flip the toggle after the chat finishes
if show_docs and st.session_state.last_docs:
    with st.container():
        st.write("---") # Visual separator
        with st.expander("Source Reviews Used"):
            for i, doc in enumerate(st.session_state.last_docs):
                st.markdown(f"**Source {i+1}** (Rating: {doc.metadata.get('rating')}/5)")
                st.caption(doc.page_content)