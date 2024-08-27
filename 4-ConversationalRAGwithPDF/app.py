##RAG Q&A Conversation with PDF Including Chat History
import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##Streamlit steup
st.title("Conversational RAG with PDF Uplaods and Chat History")
st.write("Uplaod PDF and chat with content")

##Input Groq API key    
api_key = st.text_input("Enter your Groq API key:", type ='password')
##Check if groq API key provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
session_id = st.text_input("Session_ID", value= "default_session")


##Statefully manage the chat history 
if "store" not in st.session_state:
    st.session_state.store = {}
uploded_files = st.file_uploader("Choose a PDF file", type="pdf",accept_multiple_files=True)

##Process uploaded file 
if uploded_files:
    documents = []
    for uploded_file in uploded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploded_file.getvalue())
            file_name = uploded_file.name
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =5000, chunk_overlap=500)
    final_docs=text_splitter.split_documents(documents)
    vectors = Chroma.from_documents(final_docs,embeddings)
    retreiver = vectors.as_retriever()

    ###Conversational Prompt
    contextualize_q_system_prompt = (
        "Given a chat history and latest user question"
        "which might reference to context in the chat history"
        "formulate a standalone question which can be understood"
        "without chat history. DO NOT answer the question,"
        "just formulate it as needed otherwise return it as is"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm,retreiver,contextualize_q_prompt)


    ###Answer Question Prompt 

    system_prompt = (
        "You are an assistant for question-answering tasks."
        "use the folloaing pieces of retrieved context to answer "
        "the question. If you do not know the answers, say that you"
        "do not know. Use three sentences maximum and keep the"
        "answer consise"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store [session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    coversational_rag_chain = RunnableWithMessageHistory(rag_chain,
                                                    get_session_history, 
                                                    input_messages_key="input",
                                                    history_messages_key="chat_history",
                                                    output_messages_key="answer",)
    user_input = st.text_input("Your Question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = coversational_rag_chain.invoke(
            {"input":user_input}, config={"configurable":{"session_id":session_id}},
        )
        st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History;", session_history.messages)
else:
    st.warning("Please Enter API key and uplaod documnets")   
