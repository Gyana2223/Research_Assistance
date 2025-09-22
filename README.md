📚 Research Assistant – (RAG + LLM + Streamlit)
🚀 Project Overview
This project is a Streamlit-based application built with LangChain, HuggingFace embeddings, and Chroma vector database. The app allows users to:
    Upload PDF research papers or documents  
    Ask natural language questions about the content  
    Get context-aware answers based on the uploaded documents  
    Maintain chat history for continuity across questions
The system is enhanced with retrieval-augmented generation (RAG) and is tailored for research-related tasks, ensuring structured, evidence-based, and academically styled responses.

🛠️ Features

    📂 Upload multiple PDF files  
    🔎 Chunk documents using RecursiveCharacterTextSplitter  
    📊 Store embeddings with Chroma for efficient retrieval  
    💬 Conversational Q&A with history-aware retriever  
    🤖 Multiple ML/LLM support (Groq API + HuggingFace embeddings)  
    🎯 Responses optimized for research writing and academic clarity

🧩 Tech Stack

    Python  
    Streamlit – frontend interface  
    LangChain – chaining and orchestration  
    HuggingFace (MiniLM-L6-v2) – embeddings  
    Chroma – vector store for retrieval  
    Groq LLM (Gemma2-9b-it) – language model backend  
    dotenv – environment variable management

⚙️ Installation & Setup
1️⃣ Start the process with creating a folder

2️⃣ Create and activate virtual environment
conda create -n research_assistant python=3.10 -y
conda activate research_assistant

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Setup environment variables

Create a .env file in the root folder and add:

HF_TOKEN=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key

5️⃣ Run the Streamlit app
streamlit run app.py

📖 Usage

    Launch the app → open in your browser.    
    Enter your Groq API key in the sidebar.   
    Upload one or more PDF files.    
    Ask your research-related questions in the text box.    
    Get structured, contextual answers with maintained chat history.
