from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import Tool
import os, json
import requests

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API")

llm = ChatOpenAI(model_name="gpt-4o-mini",
                 temperature=0,
                 max_tokens=None
                 )

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=20,
                                          length_function=len,
                                          is_separator_regex=False)


class WeatherForecast:
    name = "weather_tool"
    description = """Useful when the user has a query related to weather, temperature or atmosphere. Input to this tool  json  object which will\
a stringify object.This json object will have three keys named as "location", "days" and "query" and their values will be provided\
    by the user never assume a value if user does not provide any value then assign NaN value to that particular key.\
    For example, if the users asks "What is the temperature, in london, for coming 7 days"  the tool input will look like this\
    :("location":"london", "days": 7, "query": "What is the temperature"). If the user does not provide any of the input values replace it with NAN.
                   """

    def __init__(self, embeddings=embeddings, splitter=splitter, llm=llm):
        self.llm = llm
        self.embeddings = embeddings
        self.splitter = splitter
        self.url = None
        self.headers = None

    def fetch_weather_information(self, query):
        user = json.loads(query)
        location, days, query = user["location"], user["days"], user['query']
        location = str(location).strip().lower()
        days = str(days).strip().lower()
        query = str(query).strip().lower()
        print(location, days, query)
        if location == "nan":
            return "Enter your location"
        if days in {"nan", "none", ""}:
            return "Enter the time duration"
        if query in {"nan", "none", ""}:
            return "Do you have specific query"
        url = f"https://weatherapi-com.p.rapidapi.com/forecast.json?q={location}&days={days}"
        headers = {"X-RapidAPI-Key": "eab1e658b8mshd8d7503a6daa1d2p1bcae1jsndd9fe65432c8",
                   "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
                   }
        response = requests.get(url, headers=headers)
        chunks = splitter.split_text(str(response.json()))
        vector_database = FAISS.from_texts(chunks, embeddings)
        template_sample = """
                    You are an AI assistant who is polite. For general queries reply from your own knowledge base in a 
                    slightly formal manner. Answer the following based on the following:
                    {context}
                    Question:{input}
                    """
        final_prompt = ChatPromptTemplate.from_template(template_sample)
        retriever = vector_database.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        chain = create_stuff_documents_chain(llm, prompt=final_prompt)
        retrieval_chain = create_retrieval_chain(retriever, chain)
        response = retrieval_chain.invoke({"input": query})
        return response["answer"]


tool3 = WeatherForecast()
tool_forecast = Tool.from_function(
    name=tool3.name,
    description=tool3.description,
    func=tool3.fetch_weather_information,
    return_direct=True
)
# print(tool3.fetch_weather_information(json.dumps({"location": 'new york',"query": "what is the temperature"})))