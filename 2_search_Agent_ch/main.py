#  First stage of react agent, langchain + react prompt

from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

#  Either we can use pydantic or .with_output_structure
from schema import AgentResponse, Source

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")

output_parsers = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tools", "tool_names", "format_instructions", "input", "agent_scratchpad"],
).partial(format_instructions=output_parsers.get_format_instructions())

# Create the agent: Provide the llm, tools and the prompt
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_format_instructions,
)

# Agent exectutor is responsible to execute the agents and call the required tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

extract_output = RunnableLambda(
    lambda x : x["output"]
)

parse_output = RunnableLambda(
    lambda x : output_parsers.parse(x)
)
chain = agent_executor | extract_output | parse_output


def main():
    result = chain.invoke(
        {
            "input": "Search for 3 latest joobs opening for an AI Engineer in India or remote"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
