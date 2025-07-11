# 📺 YouTube Transcript Chatbot

A simple **Streamlit app** that lets you **chat with any YouTube video’s transcript** using **Google Generative AI** and **LangChain**.  
It automatically extracts the transcript, splits it into chunks, stores it in a local FAISS vector database, and answers your questions conversationally.

---

## ✨ Features

- ✅ Load any YouTube video by URL
- ✅ Automatic transcript extraction and chunking
- ✅ Store embeddings with FAISS vector database
- ✅ Chat with the video using Gemini AI
- ✅ Reuse existing video embeddings for faster access

---

## ⚙️ Requirements

* aiohappyeyeballs==2.6.1
* aiohttp==3.11.16
* aiosignal==1.3.2
* altair==5.5.0
* annotated-types==0.7.0
* anyio==4.9.0
* attrs==25.3.0
* blinker==1.9.0
* cachetools==5.5.2
* certifi==2025.1.31
* charset-normalizer==3.4.1
* click==8.1.8
* colorama==0.4.6
* dataclasses-json==0.6.7
* defusedxml==0.7.1
* faiss-cpu==1.10.0
* filetype==1.2.0
* frozenlist==1.5.0
* gitdb==4.0.12
* GitPython==3.1.44
* google-ai-generativelanguage==0.6.17
* google-api-core==2.24.2
* google-api-python-client==2.166.0
* google-auth==2.38.0
* google-auth-httplib2==0.2.0
* google-generativeai==0.8.4
* googleapis-common-protos==1.69.2
* greenlet==3.1.1
* grpcio==1.71.0
* grpcio-status==1.71.0
* h11==0.14.0
* httpcore==1.0.7
* httplib2==0.22.0
* httpx==0.28.1
* httpx-sse==0.4.0
* idna==3.10
* Jinja2==3.1.6
* jsonpatch==1.33
* jsonpointer==3.0.0
* jsonschema==4.23.0
* jsonschema-specifications==2024.10.1
* langchain==0.3.23
* langchain-community==0.3.21
* langchain-core==0.3.51
* langchain-google-genai==2.1.2
* langchain-text-splitters==0.3.8
* langsmith==0.3.25
* MarkupSafe==3.0.2
* marshmallow==3.26.1
* multidict==6.3.2
* mypy-extensions==1.0.0
* narwhals==1.34.0
* numpy==2.2.4
* orjson==3.10.16
* packaging==24.2
* pandas==2.2.3
* pillow==11.1.0
* propcache==0.3.1
* proto-plus==1.26.1
* protobuf==5.29.4
* pyarrow==19.0.1
* pyasn1==0.6.1
* pyasn1_modules==0.4.2
* pydantic==2.11.2
* pydantic-settings==2.8.1
* pydantic_core==2.33.1
* pydeck==0.9.1
* pyparsing==3.2.3
* python-dateutil==2.9.0.post0
* python-dotenv==1.1.0
* pytz==2025.2
* PyYAML==6.0.2
* referencing==0.36.2
* requests==2.32.3
* requests-toolbelt==1.0.0
* rpds-py==0.24.0
* rsa==4.9
* six==1.17.0
* smmap==5.0.2
* sniffio==1.3.1
* SQLAlchemy==2.0.40
* streamlit==1.44.1
* tenacity==9.1.2
* toml==0.10.2
* tornado==6.4.2
* tqdm==4.67.1
* typing-inspect==0.9.0
* typing-inspection==0.4.0
* typing_extensions==4.13.1
* tzdata==2025.2
* uritemplate==4.1.1
* urllib3==2.3.0
* watchdog==6.0.0
* yarl==1.19.0
* youtube-transcript-api==1.0.3
* zstandard==0.23.0

## 3️⃣ Install dependencies
``pip install -r requirements.txt``

## 🚀 Run the App
``streamlit run app.py``

### ✅ Load New YouTube Video

* Paste a YouTube URL.

* The app will download the transcript, chunk it, embed it, and save it locally in the faiss_index folder.

* Enter your question and get an answer.

### ✅ Chat with Existing Video

* If you’ve already loaded a video, you can reuse the saved embeddings to chat instantly.

### ✅ Exit

* End your session.


## 🧩 How It Works

### 🔹 `create_vector_db_from_youtube(url)`

- Loads the YouTube transcript using **YoutubeLoader**
- Splits it into chunks using **RecursiveCharacterTextSplitter**
- Embeds chunks with **GoogleGenerativeAIEmbeddings**
- Stores embeddings locally using **FAISS**

### 🔹 `get_response_from_query(db, query, k=8)`

- Loads the FAISS vector DB if needed
- Uses **ConversationalRetrievalChain** with **Gemini**
- Retrieves top relevant chunks
- Generates a conversational response

---

## 🗒️ Notes

✅ **Persistence:** Saves embeddings locally in `faiss_index/` for reuse.  
✅ **Security:** Uses `allow_dangerous_deserialization=True` when loading FAISS — use only with trusted data.  
✅ **API Costs:** Using Google Generative AI may incur usage fees.
