from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")
Hugging_face_Api = os.getenv("Huggingface_Api_Key")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
new_vector_store = FAISS.load_local("Vectordb", embeddings, allow_dangerous_deserialization=True)

llm = ChatOpenAI(model_name="gpt-4o-mini",
                 temperature=0,
                 max_tokens=None
                 )


@tool
def age(a: int):
    """Substract a from 2024."""
    return int(2024) - int(a)


@tool
def output_response(query):
    """Use the retrieval chain to answer the query"""
    template_sample = """
    Answer the following based on the following:
    {context}
    Question:{input}
    """
    final_prompt = ChatPromptTemplate.from_template(template_sample)
    retriever = new_vector_store.as_retriever(search_type="mmr",  search_kwargs={"k": 1})
    chain = create_stuff_documents_chain(llm, prompt=final_prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

Toolkit = [age, output_response]