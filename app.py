import os
import re # Penting untuk sanitasi nama file
import streamlit as st
import tempfile

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- KONFIGURASI AWAL ---
# Mematikan parallel processing tokenizer agar tidak konflik
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Config & Title
st.set_page_config(page_title="DocuMind RAG", layout="wide")
st.title("🤖 DocuMind: Chat with PDF (Hybrid Architecture)")
st.caption("Powered by: Groq (Llama-3.1) + HuggingFace Local Embeddings")

load_dotenv()

# Validasi Key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ Error: API Key Groq tidak ditemukan di .env")
    st.stop()

# --- 2. CACHING RESOURCES ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- 3. SIDEBAR: UPLOAD PDF ---
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")
    
    # Tombol Reset Manual
    if st.button("Clear Data"):
        st.session_state.clear() # Bersihkan semua state
        st.rerun()

# --- 4. PROCESSING LOGIC (CORE FIX) ---
if uploaded_file:
    # 1. Buat ID Unik (Nama + Size)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # 2. Inisialisasi State Awal jika belum ada
    if "last_upload_id" not in st.session_state:
        st.session_state["last_upload_id"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 3. Cek apakah file beda dengan ingatan terakhir?
    if st.session_state["last_upload_id"] != file_id:
        
        # UI Feedback: Spinner berjalan real-time
        with st.spinner(f"🚀 Memproses dokumen baru: {uploaded_file.name}..."):
            try:
                # A. BERSIHKAN MEMORI LAMA
                if "vectorstore" in st.session_state:
                    del st.session_state["vectorstore"]
                
                # HAPUS CHAT HISTORY LAMA (PENTING!)
                st.session_state["messages"] = [] 
                
                # B. Simpan file sementara
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # C. Load & Split
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # D. Embedding & Chroma Setup
                embeddings = load_embedding_model() 
                
                # Sanitasi Nama Collection
                clean_filename = re.sub(r'[^a-zA-Z0-9_]', '_', uploaded_file.name)
                collection_name = f"col_{clean_filename[:50]}" 

                # Buat Vectorstore Baru
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings,
                    collection_name=collection_name 
                )
                
                # E. Update Session State
                st.session_state["vectorstore"] = vectorstore
                st.session_state["last_upload_id"] = file_id
                
                # Cleanup Temp File
                os.remove(tmp_path)
                
                # Notifikasi Sukses
                st.success(f"✅ Database Siap: {collection_name}")
                
                # F. RERUN - Memaksa UI refresh total (Input text akan bersih)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan sistem: {e}")

    else:
        # Jika file sama, berikan indikator pasif
        st.sidebar.success(f"Active: {uploaded_file.name}")

# --- 5. CHAT INTERFACE (UPDATED: FIX INPUT CLEARING) ---
if "vectorstore" in st.session_state:
    
    # 1. Tampilkan History Chat (Agar chat tidak hilang saat input baru)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Setup Chain
    vectorstore = st.session_state["vectorstore"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    template = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen PDF yang diunggah.
    
    Konteks Dokumen:
    {context}
    
    Pertanyaan User: {question}
    
    Instruksi:
    1. Jawablah berdasarkan konteks di atas.
    2. Gunakan Bahasa Indonesia yang sopan dan profesional.
    3. Jika jawaban tidak ada di konteks, katakan: "Maaf, informasi tersebut tidak ditemukan dalam dokumen ini."
    
    Jawaban:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 3. Input User (MENGGUNAKAN st.chat_input)
    # Ini solusi agar text otomatis hilang setelah enter
    if user_input := st.chat_input("Tanya sesuatu tentang dokumen ini..."):
        
        # Tampilkan pertanyaan user
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        relevant_docs = retriever.invoke(user_input)

        # Proses Jawaban
        with st.chat_message("assistant"):
            with st.spinner("Sedang berpikir..."):
                response = rag_chain.invoke(user_input)
                st.markdown(response)

        # FITUR BARU: Tampilkan Sumber (Evidence)
        with st.expander("📚 Referensi Dokumen (Source)"):
            for i, doc in enumerate(relevant_docs):
                # Ambil metadata halaman jika ada
                page_num = doc.metadata.get('page', 'Unknown')
                source_file = doc.metadata.get('source', 'Unknown')
                
                st.markdown(f"**📄 Sumber {i+1} (Page {page_num}):**")
                st.caption(f"_{doc.page_content[:200]}..._") # Tampilkan 200 karakter pertama
                st.divider()
        
        # Simpan jawaban bot ke history
        st.session_state["messages"].append({"role": "assistant", "content": response})

else:
    st.info("👈 Silakan upload file PDF di sidebar kiri untuk memulai.")