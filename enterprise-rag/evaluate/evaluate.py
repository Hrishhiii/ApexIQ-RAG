from dotenv import load_dotenv
load_dotenv()

import sys
import os

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
    get_source_info,
)

from test_datasets import test_cases

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


print("Setting up RAG pipeline...")

documents = load_documents()
chunks = split_documents(documents)

vector_store = get_or_create_vector_store(chunks)
bm25_retriever = create_bm25_retriever(chunks)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

evaluator_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=2048,
)


run_config = RunConfig(
    max_workers=1,
    timeout=300,
    max_retries=2
)

# Let RAGAS use this LLM for evaluation
evaluator_llm = LangchainLLMWrapper(
    evaluator_model,
    run_config=run_config
)


def get_answer_and_context(query):
    vector_docs = retrieve_documents(query, vector_store)

    bm25_docs = retrieve_bm25_documents(
        query,
        bm25_retriever
    )

    final_docs = reciprocal_rank_fusion(
        vector_docs,
        bm25_docs
    )

    reranked_docs = reranking_documents(
        query,
        final_docs
    )

    contexts = [
        doc.page_content
        for doc in reranked_docs
    ]

    context_str = "\n\n".join([
        f"[Source: {get_source_info(doc)[0]}, "
        f"Page: {get_source_info(doc)[1]}]\n"
        f"{doc.page_content}"
        for doc in reranked_docs
    ])

    prompt_template = (
        "You are an AI assistant that provides accurate and concise "
        "answers to questions based on the provided context.\n\n"
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

    answer = llm.invoke(formatted_prompt).content

    return answer, contexts


print(
    f"\nRunning pipeline on {len(test_cases)} questions...\n"
)

dataset = []

for i, case in enumerate(test_cases, start=1):

    print(f"Processing question {i}...")

    answer, contexts = get_answer_and_context(
        case["question"]
    )

    dataset.append({
        "user_input": case["question"],
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": case["reference"],
    })


evaluation_dataset = EvaluationDataset.from_list(
    dataset
)

print("\nStarting RAGAS evaluation...\n")

result = evaluate(
    evaluation_dataset,
    llm=evaluator_llm,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
    ],
    run_config=run_config
)


print("\n" + "=" * 60)
print("RAGAS EVALUATION RESULTS")
print("=" * 60)

print(result)


df = result.to_pandas()

print("\nDetailed results:")
print(df[
    [
        "context_recall",
        "faithfulness",
        "factual_correctness(mode=f1)"
    ]
])