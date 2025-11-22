📄🔍 RAG-based Document Q&A (Offline)

A fully **offline**, **privacy-friendly** Question & Answer system powered by **Retrieval-Augmented Generation (RAG)** using:

- **Ollama** for local LLM inference  
- **ChromaDB** for blazing-fast vector search  
- **Sentence-Transformers** for high-quality embeddings  

Ask questions about your documents **without sending anything to the cloud**.

---

## 🚀 Features

- 🔐 **100% Offline** — All processing happens locally  
- 📁 **Multi-Document Support** — Load PDFs, text files, or knowledge dumps  
- 🧠 **RAG Pipeline** — Retrieve relevant chunks + generate precise answers  
- ⚡ **Fast Vector Search** with ChromaDB  
- 📦 **Simple Setup**, minimal dependencies  
- 🛠️ Works on CPU and GPU depending on your Ollama model  

---

## 🧱 Architecture Overview

```

Documents → Chunking → Embeddings (Sentence-Transformers)
↓
ChromaDB (Vector Store)
↓
Query → Retrieve Top-k Chunks → Ollama LLM → Answer

```

This hybrid approach ensures answers are accurate, grounded, and private.


## 📥 Installation

### 1️⃣ Install Ollama  
Download from: https://ollama.com/download  
Example model (customizable):  
```

ollama pull llama3

```

### 2️⃣ Install Python Dependencies  
```

pip install chromadb sentence-transformers

```

### 3️⃣ Run the App  
```

python main.py

`📚 How It Works

1. **Load documents** → Automatically chunked and embedded  
2. **Store vectors** in ChromaDB  
3. **Ask a question** → System fetches best-matching chunks  
4. **Ollama generates an answer** grounded in your documents  

No internet. No cloud APIs. Complete privacy.

📝 Example Query

**You ask:**  
> “Summarize the key points from Chapter 4.”

**The system:**  
- Finds relevant embeddings  
- Retrieves the top chunks  
- Uses Ollama to generate a grounded, contextual answer  


## 📂 Project Structure
📦 rag-doc-qa
┣ 📁 data/            # Your documents
┣ 📁 db/              # Local ChromaDB store
┣ 📜 app.py           # Main RAG interface
┣ 📜 ingest.py         # Embedding + ingestion scripts
┗ 📜 README.md

🔮 Future Enhancements
- 🖥️ Web UI (Streamlit / Gradio)  
- 📑 PDF smart segmentation  
- 🗂️ Document tagging & metadata search  
- 🎙️ Voice-based Q&A  

🤝 Contributions
Contributions, issues, and feature requests are welcome!  
Feel free to **open an issue** or submit a **pull request**.
⭐ Support
If you found this useful, please consider giving the repo a ⭐ on GitHub — it helps a lot!
