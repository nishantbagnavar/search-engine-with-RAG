import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType

# Load environment variables
load_dotenv()

# Set API Keys (Optional if deploying locally)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Conversational_RAG_Chatbot"

# Streamlit App
st.title("📖 Conversational RAG Chatbot with PDF & Web Search")
st.write("Upload PDFs and chat with their content, or search the web (Arxiv & Wikipedia).")

# Sidebar Section
with st.sidebar:
    # User inputs API Key
    api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")

    # File Upload Section
    uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

    # Search Mode Selection
    search_type = st.radio("🔍 Choose Search Mode:", ["PDF Retrieval", "Web Search (Arxiv/Wikipedia)"])

# Initialize LLM (Groq) only if API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.7)

    # Initialize Session State for Chat History
    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()
    elif isinstance(st.session_state.history, list):  # Fix in case of incorrect initialization
        st.session_state.history = ChatMessageHistory()


    # Function to Get Chat Session History
    def get_session_history(session_id):
        return st.session_state.history

    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Process uploaded PDFs
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp_{uploaded_file.name}"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            # Load PDF content
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Vector Store (FAISS)
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Retriever with larger k (fetches more chunks)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # History-aware question reformulation prompt
        contextualize_q_system_prompt = """
        Given the chat history and latest user question, rephrase it as a standalone question.
        If it doesn't require rephrasing, return it as is.
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Question-Answering Prompt
        system_prompt = """
        You are an AI assistant. Use the retrieved context to answer the user's question.
        If you don't know, say so instead of making up an answer.

        Context:
        {context}
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # RAG Pipeline
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Conversational RAG with History
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    # Define Arxiv & Wikipedia Tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # User Input
    session_id = st.text_input("🆔 Session ID:", value="default_session")
    user_input = st.text_input("✍️ Your Question:")

    if user_input:
        # Append user input to history
        st.session_state.history.add_user_message(user_input)

        # Display user message
        with st.chat_message("user"):
            st.markdown(f"**User**: {user_input}")

        # Handle PDF retrieval
        if search_type == "PDF Retrieval" and uploaded_files:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"session_id": session_id}
            )
            # Store assistant response
            st.session_state.history.add_ai_message(response["answer"])

            # Display response
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant**: {response['answer']}")

        # Handle Web Search
        elif search_type == "Web Search (Arxiv/Wikipedia)":
            search_agent = initialize_agent(
                tools=[arxiv_tool, wiki_tool],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            response = search_agent.run(user_input)

            # Store assistant response
            st.session_state.history.add_ai_message(response)

            # Display response
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant**: {response}")

        else:
            st.warning("⚠️ Please upload PDFs before using PDF Retrieval.")

    # Display the full conversation history
    st.write("### Conversation History:")
    for msg in st.session_state.history.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(f"**User**: {msg.content}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant**: {msg.content}")

else:
    st.warning("🔑 Please enter your API key to proceed.")
