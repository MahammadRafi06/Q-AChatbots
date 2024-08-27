import validators, streamlit as st 
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader, UnstructuredURLLoader
import nltk
# nltk.download('averaged_perceptron_tagger')
## Steamlit app
st.set_page_config(page_title="Langchain Summarization of text from YT or website")
st.title("Langchain Summarization of text from YT or website")
st.subheader("Summary URL")


# Get Groq key and URL to be summarized

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
generic_url = st.text_input("URL", label_visibility="collapsed")
llm = ChatGroq(groq_api_key=groq_api_key, model= "gemma-7b-it")
prompt_template = """ provide the sumamry of following conetnt in 300 words:
content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
if st.button("Summarixe the content of the link"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
elif not validators.url(generic_url):
    st.error("Please enter a valid url")
else:
    try:
        with st.spinner("Waiting...."):
            if "youtube.com" in generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
            else:
                loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, headers ={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
            docs = loader.load()
            print(docs)
            chain = load_summarize_chain(llm,chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)
            st.success(output_summary)
    except Exception as e:
        st.exception(f"Exception:{e}")




