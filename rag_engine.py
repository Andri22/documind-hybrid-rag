import os
import sys
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_groq import ChatGroq
# GANTI INI: Kita pakai HuggingFace (Local) alih-alih Google
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Config
load_dotenv()

# Cek Key (Hanya butuh Groq sekarang)
if not os.getenv("GROQ_API_KEY"):
    print("❌ Error: GROQ_API_KEY tidak ditemukan di .env")
    sys.exit(1)

def main():
    print("=== 🤖 RAG SYSTEM (Hybrid: Local Embed + Cloud LLM) ===")
    
    pdf_file = "data.pdf" 
    
    # --- 1. PREP DATA ---
    print(f"DTO: Membaca file {pdf_file}...")
    if not os.path.exists(pdf_file):
        print(f"❌ Error: File {pdf_file} tidak ditemukan.")
        return

    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    
    # OPTIMASI FREE TIER:
    # Kita batasi chunk size agar tidak terlalu membebani
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # --- CHANGE IS HERE ---
    print("DTO: Mendownload/Load Model Local (Mungkin agak lama di awal)...")
    # Model 'all-MiniLM-L6-v2' itu kecil, cepat, dan standar industri untuk CPU
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"DTO: Membuat Vector Index untuk {len(splits)} chunks...")
    # Proses ini murni offline, jadi tidak akan kena Rate Limit API
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Ambil top 3 context saja
    
    print("✅ System Ready! (Memory Index Created Locally)")

    # Versi terbaru & Tercepat saat ini (Desember 2024/2025 standard)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    template = """Jawablah pertanyaan berdasarkan context berikut ini saja:
    
    {context}
    
    Pertanyaan: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n💬 Silakan bertanya (ketik 'exit' untuk keluar).\n")

    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue

        print("🤖 AI: ...", end="\r")
        
        try:
            response = rag_chain.invoke(query)
            print(f"\r🤖 AI: {response}\n")
            print("-" * 30)
        except Exception as e:
            # Error Handling yang lebih jelas
            if "429" in str(e):
                print("\n❌ API LIMIT GROQ REACHED. Tunggu sebentar atau coba besok.")
            else:
                print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()