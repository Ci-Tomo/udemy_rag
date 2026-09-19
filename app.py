import os

import chromadb
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="シンプルRAGアプリ", page_icon=":robot_face:")

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


@st.cache_resource
def get_openai_client(api_key):
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def scrape_article(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    joined_text = soup.get_text(separator=" ", strip=True)
    return joined_text


def chunk_text(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def vectorize_text(client, text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def index_url(client, collection, url, chunk_size, overlap):
    article_text = scrape_article(url)
    text_chunks = chunk_text(article_text, chunk_size, overlap)

    embeddings = [vectorize_text(client, chunk) for chunk in text_chunks]
    ids = [f"{url}-{i}" for i in range(len(text_chunks))]
    metadatas = [{"source": url} for _ in text_chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=text_chunks,
        metadatas=metadatas,
    )
    return len(text_chunks)


def ask_question(client, question, context):
    prompt = f"""以下の質問に以下の情報をベースにして答えてください。
    [ユーザーの質問]
    {question}

    [情報]
    {context}
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


st.title("シンプルRAGアプリ")
st.caption("ChromaDB + Streamlit + OpenAI")

with st.sidebar:
    st.header("設定")
    default_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")

    st.divider()
    st.subheader("記事の登録")
    url = st.text_input("URL", placeholder="https://example.com/article")
    chunk_size = st.number_input("chunk_size", value=400, min_value=50, step=50)
    overlap = st.number_input("overlap", value=50, min_value=0, step=10)
    top_k = st.number_input("検索件数（top_k）", value=2, min_value=1, max_value=10)

    if st.button("URLをインデックス化", use_container_width=True):
        if not api_key:
            st.error("OpenAI API Keyを入力してください")
        elif not url:
            st.error("URLを入力してください")
        else:
            with st.spinner("URLをインデックス化中..."):
                client = get_openai_client(api_key)
                collection = get_chroma_collection()
                try:
                    n_chunks = index_url(client, collection, url, chunk_size, overlap)
                    st.success(f"{n_chunks}個のチャンクを登録しました。")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

    st.divider()
    collection = get_chroma_collection()
    st.metric("登録されたチャンク数", collection.count())
    if st.button("インデックスをクリア", use_container_width=True):
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_client.delete_collection(name=COLLECTION_NAME)
        st.cache_resource.clear()
        st.rerun()

st.subheader("質問する")
question = st.text_input("質問を入力してください")

if st.button("回答する", type="primary"):
    if not api_key:
        st.error("OpenAI API Keyを入力してください")
    elif not question:
        st.error("質問を入力してください")
    elif collection.count() == 0:
        st.error("まだ記事が登録されていません。先にURLをインデックス化してください。")
    else:
        client = get_openai_client(api_key)
        with st.spinner("回答を生成中..."):
            question_vector = vectorize_text(client, question)
            results = collection.query(
                query_embeddings=[question_vector],
                n_results=min(top_k, collection.count()),
            )
            top_documents = results["documents"][0]
            context = "\n\n".join(top_documents)
            answer = ask_question(client, question, context)

        st.subheader("回答")
        st.write(answer)

        with st.expander("参照したチャンク"):
            for i, doc in enumerate(top_documents, 1):
                st.markdown(f"**チャンク{i}**")
                st.text(doc)
