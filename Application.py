from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from Tools import Toolkit
import os

load_dotenv()
Api_token = os.getenv("Api_Token")
Application = Flask(__name__)
CORS(Application)


def answer(query: str):
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                     temperature=0,
                     max_tokens=None
                     )
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, Toolkit, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=Toolkit, verbose=True)
    response = agent_executor.invoke({"input": query})
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
                return answer(data["query"])["output"]
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