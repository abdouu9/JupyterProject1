# *****************************************************************************
# But : Amélioration d'un RAG classique vers un RAG avancé
# Auteur : Étudiant
# Base fournie par : Prof. OUHMIDA Asmae
#
# T.A.F :
# 1) Comprendre le RAG classique
# 2) Implémenter une amélioration (Query Expansion + Reranking)
# 3) Comparer RAG classique vs RAG amélioré
# *****************************************************************************

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ---------------------- CONFIG STREAMLIT ----------------------
st.set_page_config(page_title="PDF-RAG (Student Version)", page_icon="📘")
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
st.title("Local PDF-RAG – Student Version")

# ---------------------- SESSION STATE ----------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "llm" not in st.session_state:
    st.session_state.llm = Ollama(model="llama3.2")


# ---------------------- TODO 1 : QUERY EXPANSION ----------------------
def expand_query(query, llm):
    """
    FIXED: Now uses LLM to generate query variations instead of calling itself.

    How it works:
    1. Ask LLM to generate 2-3 alternative phrasings of the question
    2. Use all queries to retrieve documents
    3. Combine results for better coverage
    """
    # Ask LLM to generate alternative queries
    expansion_prompt = f"""
Given this question: "{query}"

Generate 2 alternative ways to ask the same question. 
Return ONLY the questions, one per line, without numbering or explanation.
"""

    # Get expanded queries from LLM
    response = llm(expansion_prompt)

    # Parse response into list of queries
    expanded_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]

    # Always include the original query
    all_queries = [query] + expanded_queries

    # Retrieve documents for all queries
    all_docs = []
    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 5})

    for q in all_queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)

    # Remove duplicates (same document retrieved multiple times)
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_id = doc.page_content[:100]  # Use first 100 chars as identifier
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)

    return unique_docs


# ---------------------- TODO 2 : RERANKING ----------------------
def rerank_chunks(query, docs, llm, top_k=5):
    """
    FIXED: Now uses LLM to score documents instead of calling itself.

    How it works:
    1. Ask LLM to rate each document's relevance (0-10)
    2. Sort by score
    3. Return top_k most relevant documents
    """
    scored_docs = []

    for doc in docs:
        # Ask LLM to score this document
        scoring_prompt = f"""
Rate the relevance of this document to the question on a scale of 0-10.
Respond with ONLY a number between 0 and 10.

Question: {query}

Document:
{doc.page_content[:500]}...

Relevance score (0-10):"""

        try:
            score_str = llm(scoring_prompt).strip()
            # Extract just the number
            score = float(score_str.split()[0])
            scored_docs.append((score, doc))
        except:
            # If parsing fails, give it a neutral score
            scored_docs.append((5.0, doc))

    # Sort by score (descending)
    scored_docs.sort(reverse=True, key=lambda x: x[0])

    # Return top_k documents
    return [doc for score, doc in scored_docs[:top_k]]


# ---------------------- PDF PROCESSING ----------------------
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        st.session_state.db = Chroma.from_documents(texts, embeddings)

        st.success("PDF processed successfully!")

# ---------------------- QUESTION / ANSWER ----------------------
query = st.chat_input("Ask a question about the PDF:")

if query and st.session_state.db:
    with st.spinner("Generating answer..."):
        # OPTION 1: Classic RAG (no expansion, no reranking)
        # retriever = st.session_state.db.as_retriever(search_kwargs={"k": 5})
        # selected_docs = retriever.get_relevant_documents(query)

        # OPTION 2: Advanced RAG (with expansion + reranking)
        # Step 1: Expand query to get more documents
        expanded_docs = expand_query(query, st.session_state.llm)

        # Step 2: Rerank to get best documents
        selected_docs = rerank_chunks(query, expanded_docs, st.session_state.llm, top_k=5)

        # ---------- FINAL PROMPT ----------
        context = "\n\n".join([doc.page_content for doc in selected_docs])

        prompt = f"""
You must answer ONLY using the context below.
If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}
"""

        answer = st.session_state.llm(prompt)

        st.write("### Answer")
        st.write(answer)

        st.write("### Sources")
        for doc in selected_docs:
            st.write(f"- {doc.metadata['source']} (page {doc.metadata.get('page', '?')})")