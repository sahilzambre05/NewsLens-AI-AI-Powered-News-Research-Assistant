import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQAWithSourcesChain

load_dotenv()

# -------------------- Configuration --------------------

st.set_page_config(
    page_title="RockyBot - News Research",
    page_icon="📈",
    layout="wide",
)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

INDEX_PATH = "gemini_faiss_index"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "GOOGLE_API_KEY is missing. Add your Gemini API key "
        "to the .env file and restart the app."
    )
    st.stop()

# Initialize Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key,
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.2,
    google_api_key=api_key,
)

# -------------------- URL Input --------------------

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"Article URL {i + 1}", key=f"url_{i}")

    if url.strip():
        urls.append(url.strip())

process_clicked = st.sidebar.button(
    "Process URLs",
    type="primary",
)

# -------------------- Process Articles --------------------

if process_clicked:
    if not urls:
        st.warning("Please enter at least one article URL.")
    else:
        try:
            with st.spinner("Loading news articles..."):
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

            if not data:
                st.error("No article content was extracted. Check the URLs.")
            else:
                with st.spinner("Splitting articles into chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ".", " ", ""],
                    )
                    docs = text_splitter.split_documents(data)

                if not docs:
                    st.error("No text chunks were created.")
                else:
                    with st.spinner("Creating Gemini embeddings..."):
                        if (
                            os.path.exists(
                                os.path.join(INDEX_PATH, "index.faiss")
                            )
                            and os.path.exists(
                                os.path.join(INDEX_PATH, "index.pkl")
                            )
                        ):
                            # Reuse the existing index and add new articles
                            vectorstore = FAISS.load_local(
                                INDEX_PATH,
                                embeddings,
                                allow_dangerous_deserialization=True,
                            )
                            vectorstore.add_documents(docs)
                        else:
                            # Create a new index
                            vectorstore = FAISS.from_documents(
                                docs,
                                embeddings,
                            )

                        # Save the index for future app sessions
                        vectorstore.save_local(INDEX_PATH)

                    st.success(
                        f"Processed {len(urls)} URL(s) and saved "
                        f"{len(docs)} text chunks."
                    )

        except Exception as e:
            st.error(f"Could not process the articles: {e}")

# -------------------- Ask Questions --------------------

st.subheader("Ask Questions About Your News")

query = st.text_input(
    "Question:",
    placeholder="What are the key takeaways from these articles?",
)

if st.button("Get Answer"):
    index_exists = (
        os.path.exists(os.path.join(INDEX_PATH, "index.faiss"))
        and os.path.exists(os.path.join(INDEX_PATH, "index.pkl"))
    )

    if not query.strip():
        st.warning("Please enter a question.")
    elif not index_exists:
        st.warning("Please process article URLs or add your saved FAISS index.")
    else:
        try:
            with st.spinner("Searching articles and generating an answer..."):
                vectorstore = FAISS.load_local(
                    INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )

                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    ),
                )

                result = chain.invoke({"question": query})

            st.subheader("Answer")
            st.write(
                result.get("answer", "No answer was generated.")
            )

            sources = result.get("sources", "")

            if sources:
                st.subheader("Sources")
                for source in sources.split("\n"):
                    if source.strip():
                        st.write(source.strip())

        except Exception as e:
            st.error(f"Could not answer your question: {e}")
