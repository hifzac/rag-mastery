import streamlit as st

from vector_store import load_index
from retriever import retrieve
from prompt_builder import build_prompt
from llm import generate_response
from config import INDEX_PATH, METADATA_PATH, TOP_K

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="ServiceNow RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.block-container{
    padding-top:2rem;
    max-width:1100px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load FAISS Once
# --------------------------------------------------

@st.cache_resource
def load_vector_database():
    return load_index(INDEX_PATH, METADATA_PATH)

index, metadata = load_vector_database()

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 ServiceNow RAG")

    st.markdown("---")

    st.subheader("AI Stack")

    st.markdown("""
- 📄 PDF Loader
- ✂️ Chunking
- 🧠 Sentence Transformers
- 🔍 FAISS
- 🤖 Groq Llama
- 💬 Conversation Memory
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 ServiceNow AI Assistant")

st.caption("Ask questions about your ServiceNow documentation.")

# --------------------------------------------------
# Display Previous Chat
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            with st.expander("📄 Sources"):

                for page in message["sources"]:
                    st.write(f"Page {page}")

# --------------------------------------------------
# User Input
# --------------------------------------------------

query = st.chat_input("Ask your question...")

if query:

    # Display user message

    with st.chat_message("user"):
        st.markdown(query)

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Retrieve relevant chunks

    with st.spinner("Searching documentation..."):

        results = retrieve(
            query=query,
            index=index,
            metadata=metadata,
            top_k=TOP_K
        )

        prompt = build_prompt(
            query=query,
            retrieved_chunks=results
        )

        pages = sorted(
            set(chunk["page_number"] for chunk in results)
        )

    # Stream Assistant Response

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        for chunk in generate_response(
            prompt=prompt,
            memory=st.session_state.messages
        ):

            full_response += chunk

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        with st.expander("📄 Sources"):

            for page in pages:
                st.write(f"Page {page}")

    # Save Assistant Response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": pages
        }
    )