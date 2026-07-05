import streamlit as st

from vector_store import load_index
from bm25_store import load_bm25
from hybrid_retriever import hybrid_search
from reranker import rerank
from prompt_builder import build_prompt
from llm import generate_response
from query_rewriter import rewrite_query

from config import (
    INDEX_PATH,
    METADATA_PATH,
    BM25_PATH,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K
)

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
# Load Vector Database
# --------------------------------------------------

@st.cache_resource
def load_databases():

    index, metadata = load_index(
        INDEX_PATH,
        METADATA_PATH
    )

    bm25 = load_bm25(BM25_PATH)

    return index, metadata, bm25


index, metadata, bm25 = load_databases()

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
- 🔎 BM25
- 🔀 Hybrid Retrieval
- 🎯 Cross Encoder Reranker
- 🤖 Groq Llama
- 💬 Conversation Memory
- 🧠 Query Rewriting
- ⚡ Streaming Responses
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 ServiceNow AI Assistant")

st.caption(
    "Ask questions about your ServiceNow documentation."
)

# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander("📄 Sources"):

                for page in message["sources"]:
                    st.write(f"Page {page}")

# --------------------------------------------------
# User Input
# --------------------------------------------------

query = st.chat_input("Ask your question...")

if query:

    # -----------------------------
    # Display User Message
    # -----------------------------

    with st.chat_message("user"):
        st.markdown(query)

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # -----------------------------
    # Retrieval Pipeline
    # -----------------------------

    with st.spinner("Searching documentation..."):

        # Step 1: Rewrite query using conversation history
        rewritten_query = rewrite_query(
            query=query,
            memory=st.session_state.messages
        )

        # Uncomment for debugging
        # st.info(f"Rewritten Query: {rewritten_query}")

        # Step 2: Hybrid Retrieval
        retrieved_chunks = hybrid_search(
            query=rewritten_query,
            index=index,
            bm25=bm25,
            metadata=metadata,
            top_k=RETRIEVAL_TOP_K
        )

        # Step 3: Rerank
        results = rerank(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=RERANK_TOP_K
        )

        # Step 4: Build Prompt
        prompt = build_prompt(
            query=query,
            retrieved_chunks=results
        )

        # Step 5: Source Pages
        pages = sorted(
            {
                chunk["page_number"]
                for chunk in results
            }
        )

    # -----------------------------
    # Generate Streaming Response
    # -----------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        for token in generate_response(
            prompt=prompt,
            memory=st.session_state.messages
        ):

            full_response += token

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        with st.expander("📄 Sources"):

            for page in pages:
                st.write(f"📄 Page {page}")

    # -----------------------------
    # Save Assistant Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": pages
        }
    ) 