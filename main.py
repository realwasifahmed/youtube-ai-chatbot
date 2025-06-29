import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_helper import create_vector_db_from_youtube, get_response_from_query

# Load environment variables from .env file
load_dotenv()

# Set the API key
def set_api_key():
    if not os.getenv("GOOGLE_API_KEY"):
        api_key = st.text_input("Please enter your Google API key:", key="api_key_input")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            st.success("API key set successfully.")
    else:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Main Streamlit app
def main():
    # Set API key
    set_api_key()

    st.title("YouTube Transcript Chatbot")

    menu = ["Load New YouTube Video", "Chat with Existing Video", "Exit"]
    choice = st.sidebar.selectbox("Select an Option", menu)

    # Use session state to persist the db (vector database)
    if 'db' not in st.session_state:
        st.session_state.db = None

    # Handle video loading
    if choice == "Load New YouTube Video":
        url = st.text_input("Enter YouTube URL:", key="youtube_url_input")
        if url:
            # Assuming the function returns a vector database
            st.session_state.db = create_vector_db_from_youtube(url)
            st.success("Video transcript loaded and vectorized!")

            # Ask the user to enter a question after the video is loaded
            query = st.text_area("Enter your question (or 'q' to go back):", key="query_input")
            if query:
                if query.lower() == 'q':
                    st.write("Going back.")
                else:
                    # Call the response function only after a valid query is entered
                    response = get_response_from_query(st.session_state.db, query)
                    st.write("Response:", response)

    # Handle chat with existing video
    elif choice == "Chat with Existing Video":
        if st.session_state.db is None and not os.path.exists("faiss_index"):
            st.error("No video loaded yet! Please load a video first.")
        else:
            if st.session_state.db is None:
                # Load the existing database if available
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

            query = st.text_input("Enter your question (or 'q' to go back):", key="existing_video_query_input")
            if query:
                if query.lower() == 'q':
                    st.write("Going back.")
                else:
                    # Call the response function only after a valid query is entered
                    response = get_response_from_query(st.session_state.db, query)
                    st.write("Response:\n", response)

    elif choice == "Exit":
        st.write("Goodbye!")

if __name__ == "__main__":
    main()
