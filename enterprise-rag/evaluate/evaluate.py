from dotenv import load_dotenv
load_dotenv()

import sys
import os
import json

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from rag import (
    load_documents,
    split_documents,
    get_or_create_vector_store,
    retrieve_documents,
    create_bm25_retriever,
    retrieve_bm25_documents,
    reciprocal_rank_fusion,
    reranking_documents,
    relevance_gate,
    get_source_info,
)

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
)

from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LOAD EVALUATION DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "eval_dataset.json")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    test_cases = json.load(f)

print(
    f"Loaded {len(test_cases)} evaluation questions "
    f"from {DATASET_PATH}"
)


# ============================================================
# SETUP RAG PIPELINE
# ============================================================

print("\nSetting up RAG pipeline...")

documents = load_documents()
chunks = split_documents(documents)

vector_store = get_or_create_vector_store(chunks)
bm25_retriever = create_bm25_retriever(chunks)


# ============================================================
# GENERATION LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# ============================================================
# EVALUATION LLM
# ============================================================

evaluator_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=2048,
)


run_config = RunConfig(
    max_workers=1,
    timeout=300,
    max_retries=2,
)

evaluator_llm = LangchainLLMWrapper(
    evaluator_model,
    run_config=run_config,
)


# ============================================================
# RAG PIPELINE FOR EVALUATION
# ============================================================

def get_answer_and_context(query):

    # --------------------------------------------------------
    # 1. VECTOR RETRIEVAL
    # --------------------------------------------------------

    vector_docs = retrieve_documents(
        query,
        vector_store
    )

    # --------------------------------------------------------
    # 2. BM25 RETRIEVAL
    # --------------------------------------------------------

    bm25_docs = retrieve_bm25_documents(
        query,
        bm25_retriever
    )

    # --------------------------------------------------------
    # 3. HYBRID RETRIEVAL - RRF
    # --------------------------------------------------------

    final_docs = reciprocal_rank_fusion(
        vector_docs,
        bm25_docs,
        top_k=10
    )

    # --------------------------------------------------------
    # 4. CROSS-ENCODER RERANKING
    #
    # reranking_documents() now returns:
    #
    # [
    #     (doc1, score1),
    #     (doc2, score2),
    #     ...
    # ]
    # --------------------------------------------------------

    reranked_docs = reranking_documents(
        query,
        final_docs,
        top_k=5
    )

    # --------------------------------------------------------
    # 5. RELEVANCE GATE
    #
    # Only documents above the relevance threshold
    # are allowed to continue to the LLM.
    # --------------------------------------------------------

    relevant_docs = relevance_gate(
        reranked_docs,
        threshold=0.50
    )

    # --------------------------------------------------------
    # 6. HANDLE RELEVANCE GATE FAILURE
    # --------------------------------------------------------

    if relevant_docs is None:

        return (
            "I'm sorry, but I couldn't find relevant information "
            "in the knowledge base to answer your question.",
            []
        )

    # --------------------------------------------------------
    # 7. EXTRACT TEXT FOR RAGAS
    # --------------------------------------------------------

    contexts = [
        doc.page_content
        for doc in relevant_docs
    ]

    # --------------------------------------------------------
    # 8. BUILD CONTEXT FOR LLM
    # --------------------------------------------------------

    context_str = "\n\n".join([
        f"[Source: {get_source_info(doc)[0]}, "
        f"Page: {get_source_info(doc)[1]}]\n"
        f"{doc.page_content}"
        for doc in relevant_docs
    ])

    # --------------------------------------------------------
    # 9. GENERATION PROMPT
    # --------------------------------------------------------

    prompt_template = (
        "You are an AI assistant that provides accurate and concise "
        "answers to questions based only on the provided context.\n\n"

        "Rules:\n"
        "1. Answer only using information present in the context.\n"
        "2. Do not make up information.\n"
        "3. If the answer is not available in the context, say:\n"
        "\"This information is not available in the provided documents.\"\n\n"

        "Context:\n{context}\n\n"

        "Question: {query}\n"

        "Answer:"
    )

    prompt = ChatPromptTemplate.from_template(
        prompt_template
    )

    formatted_prompt = prompt.invoke({
        "query": query,
        "context": context_str
    })

    answer = llm.invoke(
        formatted_prompt
    ).content

    return answer, contexts


# ============================================================
# RUN EVALUATION PIPELINE
# ============================================================

print(
    f"\nRunning pipeline on {len(test_cases)} questions...\n"
)

dataset = []

for i, case in enumerate(test_cases, start=1):

    print("=" * 60)
    print(f"Processing question {i}/{len(test_cases)}")
    print(f"Question: {case['question']}")
    print(f"Category: {case['category']}")

    answer, contexts = get_answer_and_context(
        case["question"]
    )

    print(f"Answer: {answer}")
    print(f"Contexts retrieved: {len(contexts)}")

    # --------------------------------------------------------
    # Convert JSON evaluation format into RAGAS format
    # --------------------------------------------------------

    dataset.append({
        "user_input": case["question"],
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": case["expected_answer"],
    })


# ============================================================
# CREATE RAGAS DATASET
# ============================================================

evaluation_dataset = EvaluationDataset.from_list(
    dataset
)


# ============================================================
# RUN RAGAS
# ============================================================

print("\nStarting RAGAS evaluation...\n")

result = evaluate(
    evaluation_dataset,
    llm=evaluator_llm,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
    ],
    run_config=run_config,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RAGAS EVALUATION RESULTS")
print("=" * 60)

print(result)


# ============================================================
# CONVERT RESULTS TO DATAFRAME
# ============================================================

df = result.to_pandas()

print("\nDetailed results:")

columns_to_show = [
    "user_input",
    "context_recall",
    "faithfulness",
    "factual_correctness(mode=f1)",
]

# Only show columns that actually exist
available_columns = [
    column
    for column in columns_to_show
    if column in df.columns
]

print(
    df[available_columns].to_string(index=False)
)


# ============================================================
# OPTIONAL: SHOW AVERAGE SCORES
# ============================================================

print("\n" + "=" * 60)
print("AVERAGE METRICS")
print("=" * 60)

for metric in [
    "context_recall",
    "faithfulness",
    "factual_correctness(mode=f1)",
]:
    if metric in df.columns:
        print(
            f"{metric}: "
            f"{df[metric].mean():.4f}"
        )