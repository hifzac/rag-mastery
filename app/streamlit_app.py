import streamlit as st

from vector_store import load_index
from bm25_store import load_bm25
from hybrid_retriever import hybrid_search
from reranker import rerank
from prompt_builder import build_prompt
from llm import generate_response
from query_rewriter import rewrite_query
from metrics import (
    start_timer,
    stop_timer,
    get_metrics,
    reset_metrics,
)
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
    st.subheader("⚙️ Debug")

    debug_mode = st.toggle(
        "Enable Debug Mode",
        value=False
    )

    show_rewritten_query = st.checkbox(
        "Show Rewritten Query",
        value=True,
        disabled=not debug_mode
    )

    show_retrieved_chunks = st.checkbox(
        "Show Retrieved Chunks",
        value=True,
        disabled=not debug_mode
    )

    show_reranked_chunks = st.checkbox(
        "Show Reranked Chunks",
        value=True,
        disabled=not debug_mode
    )

    show_prompt = st.checkbox(
        "Show Final Prompt",
        value=True,
        disabled=not debug_mode
    )
    show_metrics = st.checkbox(
    "Show Performance Metrics",
    value=True,
    disabled=not debug_mode
    )

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

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources"):
                for page in message["sources"]:
                    st.write(f"📄 Page {page}")

               

# --------------------------------------------------
# User Input
# --------------------------------------------------

query = st.chat_input("Ask your question...")

if query:
    reset_metrics()

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
        start_timer("Query Rewrite")

        rewritten_query = rewrite_query(
            query=query,
            memory=st.session_state.messages
        )

        stop_timer("Query Rewrite")
        if debug_mode and show_rewritten_query:

            with st.expander("🧠 Rewritten Query", expanded=False):

                st.code(rewritten_query)
        # Uncomment for debugging
        # st.info(f"Rewritten Query: {rewritten_query}")

        # Step 2: Hybrid Retrieval
        start_timer("Hybrid Retrieval")

        retrieved_chunks = hybrid_search(
            query=rewritten_query,
            index=index,
            bm25=bm25,
            metadata=metadata,
            top_k=RETRIEVAL_TOP_K,
        )

        stop_timer("Hybrid Retrieval")
        if debug_mode and show_retrieved_chunks:

            with st.expander("📄 Retrieved Chunks", expanded=False):

                for i, chunk in enumerate(retrieved_chunks, start=1):

                    st.markdown(f"### Chunk {i}")

                    st.write(f"**Chunk ID:** {chunk['chunk_id']}")
                    st.write(f"**Page:** {chunk['page_number']}")

                    st.text_area(
                        label="",
                        value=chunk["text"],
                        height=150,
                        disabled=True,
                        key=f"retrieved_{i}"
                    )

                    st.divider()

        # Step 3: Rerank
        start_timer("Cross Encoder")

        results = rerank(
            query=query,
            retrieved_chunks=retrieved_chunks,
            top_k=RERANK_TOP_K,
        )

        stop_timer("Cross Encoder")
        if debug_mode and show_reranked_chunks:

            with st.expander("🎯 Reranked Chunks", expanded=False):

                for i, chunk in enumerate(results, start=1):

                    st.markdown(f"### Rank {i}")

                    st.write(f"**Chunk ID:** {chunk['chunk_id']}")
                    st.write(f"**Page:** {chunk['page_number']}")

                    st.text_area(
                        label="",
                        value=chunk["text"],
                        height=150,
                        disabled=True,
                        key=f"reranked_{i}"
                    )

                    st.divider()
        # Step 4: Build Prompt
        start_timer("Prompt Builder")

        prompt = build_prompt(
            query=query,
            retrieved_chunks=results,
        )

        stop_timer("Prompt Builder")
        if debug_mode and show_prompt:

            with st.expander("📝 Final Prompt", expanded=False):

                st.code(prompt)
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
        start_timer("Groq LLM")

        for token in generate_response(
            prompt=prompt,
            memory=st.session_state.messages
        ):

            full_response += token

            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        stop_timer("Groq LLM")


        with st.expander("📄 Sources"):
            
        
            for page in pages:
                st.write(f"📄 Page {page}")
        if debug_mode and show_metrics:

                metrics = get_metrics()

                with st.expander("⚡ Pipeline Performance", expanded=False):

                    col1, col2 = st.columns(2)

                    items = list(metrics.items())

                    half = (len(items) + 1) // 2

                    for stage, duration in items[:half]:
                        with col1:
                            st.metric(stage, f"{duration:.3f} sec")

                    for stage, duration in items[half:]:
                        with col2:
                            st.metric(stage, f"{duration:.3f} sec")

                    st.divider()

                    st.metric(
                        "Total Pipeline",
                        f"{sum(metrics.values()):.3f} sec"
                    )
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