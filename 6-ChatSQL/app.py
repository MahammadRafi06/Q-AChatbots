import streamlit as st 
from mysql import connector
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq


st.set_page_config (page_title="LangChain: Chat with SQL DB")
st.title("LangChain: Chat with SQL DB")

INJECTION_WARNING = """"
                    SQL agent can be vulnarable to prompt injection. Use a DB role with limite
                    """

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_options = ["Connect to your my SQL Database","Use SQLite 3 Database- Student.db"]

selected_ops = st.sidebar.radio(label="Choose the db you want to chat",options=radio_options)

if radio_options.index(selected_ops) ==0:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provider MYSQL Host")
    mysql_user = st.sidebar.text_input("Provider MYSQL User")
    mysql_password = st.sidebar.text_input("Provider MYSQL password", type="password")
    mysql_db = st.sidebar.text_input("Provider MYSQL Database")
else:
      db_uri = LOCALDB
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not db_uri:
    st.info("Please select database information")

if not api_key:
    st.info("Please select api key information")

llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri == LOCALDB:
       db_filepath = (Path(__file__).parent/"student.db").absolute()
       print(db_filepath)
       creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
       return SQLDatabase(create_engine("sqlite://", creator=creator))
    elif db_uri == MYSQL:
        print(f"{mysql_host} {mysql_user} {mysql_password} {mysql_db}")
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all details to connect to SQL Database")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri == MYSQL:
   db =  configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db = configure_db(db_uri)   

toolkit = SQLDatabaseToolkit(db=db,llm=llm)

agent = create_sql_agent(llm=llm,toolkit=toolkit, verbose=True, agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

if "messages" not in st.session_state or st.sidebar.button("Clear Chat history"):
    st.session_state["messages"] = [
        {"role":"assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
     
user_query = st.chat_input(placeholder="Ask anything from database?")

if user_query:
    st.session_state.messages.append({"role":"user", "content":user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistnat", "content":response})
        st.write(response)
