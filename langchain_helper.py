from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai


def create_vector_db_from_youtube(url):
  from langchain_community.document_loaders import YoutubeLoader

  loader = YoutubeLoader.from_youtube_url(url)

  try:
    transcript = loader.load()
  except Exception as e:
    raise ValueError(f"Failed to fetch transcript: {str(e)}")

  if not transcript:
    raise ValueError("Transcript is empty or not available for this video.")

  # Now continue only if transcript is valid
  vectorstore = FAISS.from_documents(transcript, GoogleGenerativeAIEmbeddings(model="models/embedding-001",))
  vectorstore.save_local("faiss_index")

  return vectorstore

def get_response_from_query(db, query, k=8):
  if db is None:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

  # Create conversation chain
  qa = ConversationalRetrievalChain.from_llm(
      llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7),
      retriever=db.as_retriever(search_kwargs={'k': k}),
  )

  result = qa.invoke({"question": query, "chat_history": []})
  return result["answer"]