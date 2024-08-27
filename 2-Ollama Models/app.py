from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 
import os
from dotenv import load_dotenv
load_dotenv()

##LangSmith Tracking
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")  ###Used for LangSmith tracking
# os.environ['LANGCHAIN_TRACING_V2'] ="true"
# os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

##Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistnat, Please respons to user questions"),
    ("user","Question:{question}")
])

def generate_response(question,engine,temparature,max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

#Title of the app
st.title("Enhanced Q&A chatbot with OpenAI")

st.sidebar.title("Settings")

engine = st.sidebar.selectbox("Select Open AI Models",["mistral","gemma2:2b","llama3.1"])
temparature = st.sidebar.slider("Temparature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)



##Main interface for user inputs
st.write("Go ahead and ask any questions")
user_input = st.text_input("You:")
if user_input:
    response = generate_response(user_input,engine,temparature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")