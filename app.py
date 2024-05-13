from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import SeleniumURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
def scrape(urls):
    loader = SeleniumURLLoader(urls= urls)
    documents = loader.load()
    return documents

st.title('Text summarizer')
with st.form(key='form1'):
    
    url = st.text_input("Web URL", placeholder="URL to be summarised", key='input')
    submit_button = st.form_submit_button(label='Summarize')

if submit_button and url:
    info = scrape([f"{url}"])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(info)
    summary_chain = load_summarize_chain(llm=llm, chain_type="stuff")
    summary = summary_chain(text_chunks)
    st.header('Summary: ')
    st.write(summary['output_text'])

