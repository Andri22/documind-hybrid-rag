import os
import sys
from dotenv import load_dotenv

# --- KONFIGURASI AWAL ---
# Mematikan parallel processing tokenizer agar tidak konflik
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Config
load_dotenv()

# Konstanta Path
CHROMA_PATH = "chroma_db"  # Folder untuk menyimpan database vektor
PDF_FILE = "data.pdf"

def get_embedding_function():
    # Menggunakan model yang ringan untuk CPU
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_split_pdf(file_path):
    print(f"📄 Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Diperbesar sedikit agar konteks kalimat utuh
        chunk_overlap=100    # Overlap diperbesar untuk menjaga kesinambungan antar chunk
    )
    splits = text_splitter.split_documents(docs)
    return splits

def initialize_vectorstore():
    """
    Logic Cerdas: Cek apakah DB sudah ada.
    Jika ADA -> Load dari disk.
    Jika TIDAK -> Buat baru dari PDF.
    """
    embedding_fn = get_embedding_function()

    # Cek apakah folder database sudah ada
    if os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH):
        print(f"💾 Memuat Vector Database yang sudah ada dari '{CHROMA_PATH}'...")
        # Load existing DB
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)
    else:
        print("🆕 Database belum ditemukan. Memulai proses Indexing baru...")
        
        if not os.path.exists(PDF_FILE):
            print(f"❌ Error: File {PDF_FILE} tidak ditemukan.")
            sys.exit(1)

        splits = load_and_split_pdf(PDF_FILE)
        
        print(f"📊 Membuat Index untuk {len(splits)} chunks... (Ini hanya sekali)")
        # Create and Save DB
        db = Chroma.from_documents(
            documents=splits, 
            embedding=embedding_fn, 
            persist_directory=CHROMA_PATH
        )
        print("✅ Database berhasil disimpan!")

    return db

def main():
    print("=== 🤖 RAG SYSTEM V2 (Persistent Storage) ===")
    
    # Validasi API Key
    if not os.path.exists(".env") or not os.getenv("GROQ_API_KEY"):
        print("❌ Warning: Pastikan .env berisi GROQ_API_KEY")

    # 1. Inisialisasi Database (Hanya load jika sudah ada)
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Setup LLM (Groq Llama 3)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 3. Prompt Engineering (Anti-Halusinasi)
    template = """Anda adalah asisten AI yang bertugas menjawab pertanyaan berdasarkan dokumen yang diberikan.
    
    ATURAN:
    1. Jawablah HANYA berdasarkan konteks berikut.
    2. Jika jawaban tidak ada di dalam konteks, katakan dengan jujur: "Maaf, informasi tersebut tidak ditemukan dalam dokumen."
    3. Jangan mengarang jawaban.
    
    Context:
    {context}
    
    Pertanyaan User: {question}
    
    Jawaban:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Chain Definition (LCEL)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n💬 System Ready. Silakan bertanya! (ketik 'exit' untuk keluar)")

    # 5. Chat Loop
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue

            print("🤖 AI: Sedang berpikir...", end="\r")
            
            # Invoke Chain
            response = rag_chain.invoke(query)
            
            # Print hasil bersih
            print(f"\r🤖 AI: {response}")
            
        except Exception as e:
            if "429" in str(e):
                print("\n❌ Rate Limit Groq Reached. Coba lagi nanti.")
            else:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()