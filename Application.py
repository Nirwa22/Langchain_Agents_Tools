from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent, AgentType
from langchain.memory import ChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from Tools.Age_calculator import tool_AgeCalculator
from Tools.QA_Chain import tool_retriever
from Tools.Weathertool import tool_forecast
import os

load_dotenv()
Api_token = os.getenv("Api_Token")
Application = Flask(__name__)
CORS(Application)

llm = ChatOpenAI(model_name="gpt-4o",
                 temperature=0
                 )
memory = ChatMessageHistory(session_id="test_session")
Toolkit = [tool_AgeCalculator, tool_retriever, tool_forecast]


memory_react_agent = ConversationBufferMemory(memory_key="chat_history")


def answer(query):
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, Toolkit, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=Toolkit, verbose=True, handle_parsing_errors=True, memory=memory_react_agent)
    response = agent_executor.invoke({"input": query})
    print(response)
    return response

@Application.route("/")
def hello_page():
    return "Home Route"


@Application.route("/get_answer", methods=["POST"])
def query():
    api = request.headers.get("Authorization")
    if api == Api_token:
        try:
            data = request.get_json()
            if data["query"]:
                return {"Output": str(answer(data["query"])["output"])}
            else:
                return {"Message": "Enter your query"}
        except Exception as e:
            return e
    elif api and api != Api_token:
        return {"Message": "Unauthorized access"}
    else:
        return {"Message": "Api key needed"}


if "__main__" == __name__:
    Application.run(debug=True)