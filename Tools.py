from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
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


def age(a: int):
    """Minus a from 2024 """
    return int(2024) - int(a)


def reply(query):
    """Reply for social and general queries"""
    template_sample = """
    """
    return "Do you have any further queries"

def output_response(query):
    template_sample = """
    You are an AI assistant who is polite. For general queries reply from your own knowledge base in a slightly 
    formal manner. Answer the following based on the following:
    {context}
    Question:{input}
    """
    final_prompt = ChatPromptTemplate.from_template(template_sample)
    retriever = new_vector_store.as_retriever(search_type="mmr",  search_kwargs={"k": 1})
    chain = create_stuff_documents_chain(llm, prompt=final_prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]


Toolkit = [Tool(
                name="Age_Calculator",
                func=age,
                description="""Only useful when user asks for their age otherwise if 
                the user just provides their birth year or birth date or birth
                month without asking their age then do not reply with their age."""
                ), Tool(
                        name="QA",
                        func=output_response,
                        description="""Useful for answering user's query regarding the vector
                        database and general queries."""
                        )]
