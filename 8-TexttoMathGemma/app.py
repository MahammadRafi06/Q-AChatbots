import streamlit as st  
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
## Steamlit app
st.set_page_config(page_title="Text to math probelm solver")
st.title("Text to math probelm solver")
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.info("Please provide groq api key")
    st.stop()
llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-it")

##Initialize the tools
wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
                    name="wikipedia",
                    func=wiki_wrapper.run,
                    description="Wikipedia tool"
                )

#Initialize math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
                    name="Calculator",
                    func=math_chain.run,
                    description="Only provider math related questions"
                )
prompt = """" Your an agent tasked to solve users mathematical questions posted as text. Logically solve the questions and sisplay the results.
Question:{question}
Answer
"""

prompt_template = PromptTemplate(input_variables=["question"], template=prompt)

###combine all tools
chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
                    name="reasoning",
                    func=chain.run,
                    description="Tool for solving logical reasoning question"
                    )

## Initialize the agents

assistant_agents = initialize_agent(
                                        tools=[wiki_tool,calculator,reasoning_tool],
                                        llm=llm,
                                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                        verbose=False,
                                        handle_parsing_error = True
                                    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content": "Hi, I am a math chatbot who can solve math questions"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

#Function to geenrate the response 


#Lets strat the interactions

Q = st.text_area("Enter question'")
if st.button("fina answer"):
    if Q:
        #st.write(Q)
        with st.spinner("Generate response"):
            st.session_state.messages.append({"role":"user", "content":Q})
            st.chat_message("user").write(Q)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agents.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistnat", "content":response})
            st.write("####Response")
            st.success(response)
    else:
        st.write("Please enter your input")