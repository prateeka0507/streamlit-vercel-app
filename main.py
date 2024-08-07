import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
import os
import datetime
import tempfile
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain_core.prompts import ChatPromptTemplate
# Load environment variables from .env file
load_dotenv()
# Access your API key
api_key = os.getenv('OPENAI_API_KEY')
# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.4,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]
)
# Initialize conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'response' not in st.session_state:
    st.session_state.response = ""
# Combine the prompt template and the model
chain = prompt | model
# Define a function to process and display the response
def process_query():
    query = st.session_state.query
    try:
        # Append the user query to the conversation history
        st.session_state.messages.append(("human", query))
        # Create the input format for the chain
        input_data = {
            "input": " ".join([f"{role}: {msg}" for role, msg in st.session_state.messages])
        }
        response = chain.invoke(input_data)
        # Append the AI response to the conversation history
        st.session_state.messages.append(("ai", response.content))
        st.session_state.response = response.content
        st.session_state.query = ""  # Clear the input field after processing
    except Exception as e:
        st.session_state.response = f"An error occurred: {str(e)}"
# Define a function to clear the conversation history
def clear_conversation():
    st.session_state.messages = []
    st.session_state.response = ""
    st.rerun()  # Rerun the script to reset the UI
# Streamlit UI
st.title("KLM-Buddy")
st.write("Ask me bro!!")
# Display the conversation history
if st.session_state.messages:
    st.write("## Conversation History")
    for role, msg in st.session_state.messages:
        st.write(f"**{role.capitalize()}:** {msg}")
# Place the input box at the bottom
st.markdown("<div style='position: fixed; bottom: 10px; width: 100%;'>", unsafe_allow_html=True)
query = st.text_input("Enter your question:", key="query", on_change=process_query)
st.markdown("</div>", unsafe_allow_html=True)
# Clear conversation button
if st.button("Clear Conversation"):
    clear_conversation()