from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_rag_pipeline(
    vector_store_path="data/vector_store",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini"
):
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda' if os.getenv('CUDA_AVAILABLE') else 'cpu'}
    )
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Load OpenAI LLM with API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    llm = ChatOpenAI(
        model=llm_model,
        openai_api_key=api_key,
        max_tokens=150,
        temperature=0.7
    )

    # Define custom prompt template
    prompt_template = """Using the provided context, generate a clear, concise, and structured answer to the question in a natural tone, as if explaining to a general audience. Do not include any instructional text or repeat the question. If the context lacks sufficient information, respond with "I don't know."

    Context:
    {context}

    Question: {question}

    Answer: """
    prompt = PromptTemplate.from_template(
        template=prompt_template
    )

    # Set up RAG pipeline
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def query_rag(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    qa_chain = load_rag_pipeline()
    question = "What is fundamental analysis?"
    answer, sources = query_rag(qa_chain, question)
    print("Answer:", answer)
    print("\nSources:")
    for doc in sources:
        print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")