from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN") # Initialize HF_TOKEN in .env file
embedddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

st.title("Welcome to Streamlit")


# api_key = os.getenv("GROQ_API_KEY")
api_key = st.text_input("Enter GROQ_API_KEY", type="password") #Enter your own API_Key

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")
    session_id = st.text_input("Session_id", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files = st.file_uploader("Choose file to upload", type='pdf', accept_multiple_files=True)


    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
                loader = PyPDFLoader(temp_pdf)
                docs = loader.load()
                documents.extend(docs)
        
        ### Split and create embedding

        text_split = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap = 300)
        split = text_split.split_documents(documents)
        vector_store = Chroma.from_documents(documents=split, embedding=embedddings)
        retriever = vector_store.as_retriever()


        ### Prompt message
        q_system_prompt = (
            "Given a chat history and a new question. Based"
            "on the presious chat and the recent question"
            "you have to answer the question asked. The answer should"
            "be most likely to be matched properly."
        )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        history_retriever = create_history_aware_retriever(llm, retriever, final_prompt)


        system_prompt = (
            "You are an AI assistant specialized in supporting research-related tasks."  
            "Provide accurate, evidence-based, and well-structured responses."  
            "Communicate in a clear, professional, and academic tone."  
            "When explaining technical or complex concepts, break them into logical steps."  
            "If the user provides incomplete input, ask clarifying questions before answering."  
            "Support responses with examples, references, or structured lists when appropriate."  
            "Avoid vague or unsupported claims; be precise and concise."  
            "Do not generate unsafe, biased, or ethically problematic content."  
            "If the information is uncertain or outside scope, clearly state the limitation."  
            "Always aim to assist in writing, analyzing, or summarizing research in a reliable manner."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )


        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)


        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Input Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("chat Message:", session_history.messages)

    else:
        st.warning("Please enter GROQ key")