from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
import os


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API")

llm = ChatOpenAI(model_name="gpt-4o-mini",
                 temperature=0,
                 max_tokens=None
                 )

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
new_vector_store = FAISS.load_local("Vectordb", embeddings, allow_dangerous_deserialization=True)


class RetrievalChain:
    name = "QA_chain"
    description = """Useful for answering user's query regarding the vector
                     database and general queries."""

    def __init__(self, vdb=new_vector_store, llm=llm):
        self.vdb = vdb
        self.llm = llm

    def retrieve(self, query):
        template_sample = """
            You are an AI assistant who is polite. For general queries reply from your own knowledge base in a slightly 
            formal manner. Answer the following based on the following:
            {context}
            Question:{input}
            """
        final_prompt = ChatPromptTemplate.from_template(template_sample)
        retriever = new_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        chain = create_stuff_documents_chain(self.llm, prompt=final_prompt)
        retrieval_chain = create_retrieval_chain(retriever, chain)
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]


tool1 = RetrievalChain()
tool_retriever = Tool.from_function(
    name=tool1.name,
    description=tool1.description,
    func=tool1.retrieve,
    return_direct=True
)
