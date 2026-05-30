import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# NEW UPDATED IMPORTS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. SETUP & CONFIGURATION
# We initialize our LLM and Embedding models here. This falls under "Model I/O".
st.set_page_config(page_title="YT Key Points Extractor", layout="wide")
st.title("📹 YouTube Video Key Points Extractor & Chat")

# Ensure API Key is available
os.environ["OPENAI_API_KEY"] = "Enter you OPENAI_API_KEY" # Replace with your actual key

# Initialize the LLM (Model I/O)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
# Initialize the Embeddings model (used for Retrieval/Vector Stores)
embeddings = OpenAIEmbeddings()

# 2. SESSION STATE MANAGEMENT (Streamlit specific)
# We use this to keep track of the vector store and chat history across UI reruns.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Helper function to manage memory for specific chat sessions."""
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]


# --- UI: LEFT SIDEBAR (Input & Extraction) ---
with st.sidebar:
    st.header("1. Extract Video Content")
    video_url = st.text_input("Paste YouTube Video URL:")
    generate_btn = st.button("Generate Summary & key Points")

    if generate_btn and video_url:
        with st.spinner("Processing video transcript..."):
            try:
                # A. RETRIEVAL (Data Loading)
                # LangChain Document Loaders fetch unstructured data from a source and convert it into text Documents.
                loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
                docs = loader.load()
                
                if not docs:
                    st.error("Could not retrieve transcript for this video.")
                    st.stop()

                # B. RETRIEVAL (Text Splitting)
                # Transcripts can be too long for an LLM's context window. 
                # We split the long document into smaller, manageable chunks.
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # C. VECTOR STORE & EMBEDDINGS (Bonus requirement)
                # We take the text chunks, turn them into vectors using OpenAIEmbeddings,
                # and store them in Chroma (an in-memory vector database) for semantic search.
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.session_state.vector_store = vectorstore
                st.success("Video processed and indexed successfully!")

                # D. PROMPTS & CHAINS (Summary Generation)
                # We define a strict prompt template instructing the LLM how to parse the text.
                summary_prompt = ChatPromptTemplate.from_template(
                    "You are an expert video analyst. Based on the following transcript chunks, "
                    "provide a concise summary of the video, followed by a bulleted list of the "
                    "top key concepts/takeaways.\n\n"
                    "Transcript:\n{context}\n\n"
                    "Output Format:\n"
                    "## Summary\n[Your summary here]\n\n"
                    "## Key Concepts\n- [Concept 1]\n- [Concept 2]"
                )
                
                # 'create_stuff_documents_chain' is a built-in LangChain Chain. 
                # It "stuffs" all the provided documents straight into the prompt template context.
                summary_chain = create_stuff_documents_chain(llm, summary_prompt)
                
                # Run the chain using the downloaded transcript text
                with st.spinner("Analyzing and summarizing..."):
                    summary_output = summary_chain.invoke({"context": docs})
                    st.session_state.summary = summary_output

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- UI: MAIN AREA (Display & Chat) ---
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Summary & Key Takeaways")
    if "summary" in st.session_state:
        st.markdown(st.session_state.summary)
    else:
        st.info("Paste a URL and click 'Generate Summary' to see results here.")

with col2:
    st.header("💬 Chat with Video")
    
    if st.session_state.vector_store is not None:
        # E. RETRIEVAL (The Retriever)
        # We turn our vector database into a 'retriever' object so LangChain knows how to query it.
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

        # F. MEMORY & CONVERSATIONAL CHAINS (History-Aware Retrieval)
        # If a user says "What did he say after that?", the retriever needs context from the chat history
        # to understand what "that" means. This prompt re-phrases the user's question.
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # This chain adjusts the search query based on chat history before hitting the vector store
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # G. THE FINAL QA CHAIN
        # This system prompt answers the user's question using ONLY the retrieved video chunks.
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context from a video transcript to answer "
            "the question. If you don't know the answer, say that you don't know.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # This master chain links the history-aware retriever to the final QA answering system.
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # H. MEMORY ENFORCEMENT
        # Wraps our RAG chain so it automatically reads/writes to our session history object.
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # UI Chat Interface
        user_session_id = "default_user_session"
        history_obj = get_session_history(user_session_id)

        # Display previous chat messages
        for msg in history_obj.messages:
            with st.chat_message(msg.type):
                st.write(msg.content)

        # Accept new user input
        if user_query := st.chat_input("Ask something about the video..."):
            with st.chat_message("human"):
                st.write(user_query)
            
            with st.chat_message("ai"):
                with st.spinner("Searching transcript..."):
                    # Invoke the entire complex LangChain system with one call
                    response = conversational_rag_chain.invoke(
                        {"input": user_query},
                        config={"configurable": {"session_id": user_session_id}}
                    )
                    st.write(response["answer"])
    else:
        st.info("The chat interface will activate once a video is successfully processed on the left.")