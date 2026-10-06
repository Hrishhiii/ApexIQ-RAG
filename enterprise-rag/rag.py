"""Core document discovery helpers for the enterprise RAG application."""
from dotenv import load_dotenv
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler


load_dotenv()
#intialize Langfuse for tracing
langfuse = get_client()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",activation_fn = torch.nn.Sigmoid())



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

def load_documents():
    documents = []
    documents_dir = os.path.join(os.path.dirname(__file__), "data", "documents")
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(documents_dir, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks =  text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def get_or_create_vector_store(chunks):
    if os.path.exists("./chroma_db"):
        print("Loading existing vectore store")
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        print("Creating new vector store")
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

@observe(name="Vector Retrieval") # Manual tracing of the vector retrieval function using Langfuse
def retrieve_documents(query, vector_store):
    documents = vector_store.similarity_search(query, k=10)
    return documents

def create_bm25_retriever(chunks):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 10
    return retriever

@observe(name="BM25 Retrieval") # Manual tracing of the BM25 retrieval function using Langfuse
def retrieve_bm25_documents(query, bm25_retriever):
    return bm25_retriever.invoke(query)


# Reciprocal Rank Fusion (RRF) implementation
def doc_key(doc):
    return (
        doc.metadata.get("source",""),
        doc.metadata.get("page",""),
        doc.page_content
    )

@observe(name="Reciprocal Rank Fusion") # Manual tracing of the RRF function using Langfuse
def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60, top_k=5):
    #contains the word and the rrf score of the document
    scores = {} 
    #contains the Document key → actual LangChain Document
    documents = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc_key(doc)
        scores[key] = scores.get(key,0) + 1/(k + rank)
        documents[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc_key(doc)
        scores[key] = scores.get(key,0) + 1/(k + rank)
        documents[key] = doc

    # Sort the documents based on their RRF scores in descending order
    ranked_keys = sorted(
        scores,
        key= scores.get,
        reverse=True
    )

    return [documents[key] for key in ranked_keys[:top_k]]

@observe(name="Reranking Documents") # Manual tracing of the reranking function using Langfuse
def reranking_documents(query, documents, top_k=5):
    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]
    scores = cross_encoder.predict(pairs)
    reranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked_docs[:top_k]

def relevance_gate(reranked_docs, threshold=0.50):
    relevant_docs = [
        doc for doc,score in reranked_docs
        if score >= threshold
    ]
    if not relevant_docs:
        return None
    
    return relevant_docs


def get_source_info(doc):
    source = doc.metadata.get("source", "N/A")
    page = doc.metadata.get("page", "N/A")

    if isinstance(page, int):
        page += 1

    return source, page

def generate_answer(context, query, source_docs, langfuse_handler):
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR policy assistant.

Answer ONLY based on the context below.

For every factual statement, include the source and page
from the context that supports the statement.

Use this format:
[Source: filename, Page: X]

If the answer is not in the context, say:
"This information is not available in the provided documents."

Context:
{context}"""),
    ("human", "{query}")
])

    formatted_prompt = prompt.invoke({"query": query, "context": context})
    answer = llm.invoke(
        formatted_prompt,
        config={"callbacks": [langfuse_handler]}
    ).content
    #Build source list for the answer(Citations)
    seen = set()
    sources = []

    for doc in source_docs:
        source, page = get_source_info(doc)
        entry = f"{source}, Page: {page}"
        if entry not in seen:
            seen.add(entry)
            sources.append(entry)
    return answer, sources

#Query-Rewriting function for RAG pipeline
contextualize_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a chat history and the latest user question,
formulate a standalone question that can be understood
without the chat history.

Rules:
1. Resolve references such as "it", "they", "this", "that", etc.
2. Preserve the user's original intent.
3. Do not answer the question.
4. Do not add information that is not present in the conversation.
5. If the question is already standalone, return it unchanged.
6. Return only the standalone question."""
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])

def requery_writing(query,history,llm,langfuse_handler):
    if not history:
        return query

    response = llm.invoke(
        contextualize_query_prompt.format_messages(
            input=query,
            chat_history=history
        ),
        config={"callbacks": [langfuse_handler]}
    )
    return response.content.strip()

@observe(name="RAG Pipeline") # Manual tracing of the RAG pipeline using Langfuse
def rag_pipeline(query, history=None, session_id=None):
    if history is None:
        history = []

    with propagate_attributes(session_id=session_id):
        #Tracing the query rewriting process using Langfuse
        langfuse_handler = CallbackHandler()

        query = requery_writing(query, history, llm, langfuse_handler)
        print("\nQuery:")
        print(query)
        # 1. LOAD DOCUMENTS
        documents = load_documents()

        if not documents:
            return "No PDF documents found in the data/documents directory."

        # 2. SPLIT DOCUMENTS INTO CHUNKS
        chunks = split_documents(documents)

        # 3. VECTOR RETRIEVAL
        vector_store = get_or_create_vector_store(chunks)

        # Retrieve top 5 documents using semantic/vector search
        vector_docs = retrieve_documents(query, vector_store)

        print("\n========== VECTOR RESULTS ==========")

        for i, doc in enumerate(vector_docs):
            print(f"\n--- Vector Result {i + 1} ---")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(doc.page_content)

        # 4. BM25 RETRIEVAL
        bm25_retriever = create_bm25_retriever(chunks)

        # Retrieve top 5 documents using keyword/BM25 search
        bm25_docs = retrieve_bm25_documents(query, bm25_retriever)

        print("\n========== BM25 RESULTS ==========")

        for i, doc in enumerate(bm25_docs):
            print(f"\n--- BM25 Result {i + 1} ---")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(doc.page_content)

        # 5. HYBRID RETRIEVAL - RRF
        # Combine Vector + BM25 rankings using
        # Reciprocal Rank Fusion
        final_docs = reciprocal_rank_fusion(
            vector_docs,
            bm25_docs,
            top_k=10
        )

        print("\n========== HYBRID RRF RESULTS ==========")

        for i, doc in enumerate(final_docs):
            print(f"\n--- RRF Result {i + 1} ---")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(doc.page_content)

        reranked_docs = reranking_documents(query, final_docs, top_k=5)

        print("\n========== RERANKED RESULTS ==========")

        for i, (doc, score) in enumerate(reranked_docs):
            print(f"\n--- Reranked Result {i + 1} ---")
            print(f"Score: {score:.4f}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(doc.page_content)

        relevant_docs = relevance_gate(reranked_docs)

        if relevant_docs is None:
            return {
            "answer": (
                "I'm sorry, but I couldn't find relevant information in the knowledge base to answer your question."
            ),
            "sources": []
        }

        # 6. CREATE CONTEXT FOR THE LLM
        # Take the final relevant documents and combine their text
        # into one context string.
        context = "\n\n".join([
        f"[Page {get_source_info(doc)[1]}]\n{doc.page_content}"
        for doc in relevant_docs
    ])

        # 7. SEND FINAL CONTEXT + QUERY TO LLM
        answer, sources = generate_answer(context, query, relevant_docs, langfuse_handler)

        return {
        "answer": answer,
        "sources": sources,
    }
