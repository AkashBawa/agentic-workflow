from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI

from langchain.agents.react.agent import create_react_agent

load_dotenv()

# @tool decorate convert this function into a LangChain tool, [ How to create custom tools ]
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    return len(text)



def main():

    #  Below way to call the tool function is deprecated, use .invoke() instead
    # print(get_text_length("Hello, world!"))
    # print(get_text_length.invoke(input="Hello, world!!"))
    print("Hello from 3-react-agent!")
    tools = [get_text_length]

    template="""
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:
    """
 
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
    llm = ChatOpenAI(temperature=0, stop=["\nObservation:", "Observation:", "Observation"])

    agent = {"input": lambda x: x["input"]} | prompt | llm

    res = agent.invoke({"input": "What is the length of the text 'Hello, world!'?"})
    print(res["text"])
if __name__ == "__main__":
    main()
