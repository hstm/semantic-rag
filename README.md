# 🦀 RAG System with Semantic Chunking

A high-performance Retrieval-Augmented Generation (RAG) system built in Rust, featuring **true semantic chunking** for superior document processing and retrieval quality.

## ✨ Features

- **🧠 Semantic Chunking**: Uses sentence embeddings and cosine similarity to create semantically coherent chunks
- **⚡ ONNX Runtime**: Fast inference with optimized ONNX models
- **🔍 Vector Search**: Powered by Qdrant for efficient similarity search
- **📄 PDF Support**: Upload and process multiple PDF documents
- **🌐 Web Interface**: Beautiful, intuitive UI for document management and querying
- **🚀 High Performance**: Built entirely in Rust for maximum speed and efficiency

## 🎯 What Makes This Special?

Unlike traditional RAG systems that split text by paragraphs or fixed character counts, this implementation uses **semantic chunking**:

1. Splits documents into sentences
2. Generates embeddings for each sentence
3. Groups sentences by semantic similarity (cosine distance)
4. Respects maximum chunk size constraints
5. Merges small chunks intelligently

This approach ensures that retrieved contexts are semantically coherent, leading to better RAG performance.

## 🛠️ Prerequisites

- **Rust** (1.70+)
- **Qdrant** vector database
- **ONNX embedding model** (e.g., all-MiniLM-L6-v2)

## 📦 Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Install Qdrant

Using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or download from [Qdrant releases](https://github.com/qdrant/qdrant/releases)

### 3. Download the Embedding Model

Download the pre-converted ONNX model or convert it yourself:

```bash
# Create models directory
mkdir -p models

# Option 1: Download pre-converted model
# Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Option 2: Convert with Optimum (requires Python)
pip install optimum[onnxruntime]  # or optimum[onnx]
pip install accelerate
optimum-cli export onnx -m sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction models/
```

Place the model at `models/model.onnx`

### 4. Clone and Build

```bash
git clone https://github.com/yourusername/rag-semantic-chunking.git
cd rag-semantic-chunking
cargo build --release
```

## 🚀 Usage

### Start the Server

```bash
cargo run --release
```

The server will start at `http://127.0.0.1:3000`

### Web Interface

Open your browser and navigate to `http://127.0.0.1:3000`

**Features:**
- 📝 **Text Input Tab**: Add documents directly as text
- 📄 **PDF Upload Tab**: Upload and process multiple PDF files
- 🔍 **Query Interface**: Ask questions and get AI-powered answers
- 🗑️ **Database Management**: Clear the database when needed

### API Endpoints

#### Add Documents (Text)
```bash
curl -X POST http://127.0.0.1:3000/add_documents \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Your document text here..."]}'
```

#### Upload PDFs
```bash
curl -X POST http://127.0.0.1:3000/upload_pdfs \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

#### Query
```bash
curl -X POST http://127.0.0.1:3000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

#### Clear Database
```bash
curl -X POST http://127.0.0.1:3000/clear_database
```

## 📊 How Semantic Chunking Works

```
Input Document
      ↓
Split into Sentences
      ↓
Generate Sentence Embeddings
      ↓
Calculate Cosine Similarity Between Adjacent Sentences
      ↓
Group Similar Sentences (threshold: 0.7)
      ↓
Respect Max Chunk Size (1000 chars)
      ↓
Merge Small Chunks (<200 chars)
      ↓
Semantically Coherent Chunks
```

**Parameters:**
- `similarity_threshold`: 0.7 (sentences with similarity > 0.7 are grouped together)
- `max_chunk_size`: 1000 characters
- `min_chunk_merge_size`: 200 characters

## 🏗️ Architecture

```
┌─────────────┐
│  Web UI     │
└──────┬──────┘
       │
┌──────▼──────────┐
│  Axum Server    │
└──────┬──────────┘
       │
┌──────▼──────────┐
│  RAG System     │
│                 │
│  ┌───────────┐  │
│  │ Semantic  │  │
│  │ Chunking  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   ONNX    │  │
│  │  Runtime  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │  Qdrant   │  │
│  │  Vector   │  │
│  │    DB     │  │
│  └───────────┘  │
└─────────────────┘
```

## 🔧 Configuration

Edit the following in `main.rs`:

```rust
// Embedding model path
let mut rag = RagSystem::new("models/model.onnx").await?;

// Qdrant connection
let qdrant_client = Qdrant::from_url("http://localhost:6334").build()?;

// Collection name
let collection_name = "documents".to_string();

// Vector dimension (all-MiniLM-L6-v2)
let vector_size = 384;

// Chunking parameters
let chunks = self.semantic_chunk_text(&doc, 0.7, 1000)?;
//                                          ^    ^
//                                          |    max_chunk_size
//                                          similarity_threshold
```

## 📝 Dependencies

```toml
[dependencies]
axum = { version = "0.7", features = ["multipart"] }
tokio = { version = "1.48.0", features = ["full"] }
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.145"
ort = { version = "2.0.0-rc.10", features = ["download-binaries"] }
qdrant-client = "1.15"
tower-http = { version = "0.6.6", features = ["cors"] }
anyhow = "1.0.100"
pdf-extract = "0.7"
regex = "1.12.2"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [ONNX Runtime](https://onnxruntime.ai/) for fast inference
- [Qdrant](https://qdrant.tech/) for vector search
- [sentence-transformers](https://www.sbert.net/) for the embedding model
- [Axum](https://github.com/tokio-rs/axum) for the web framework

## 📚 Learn More

- [What is RAG?](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
- [Sentence Transformers](https://www.sbert.net/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [ONNX Runtime Rust](https://docs.rs/ort/latest/ort/)

## 🐛 Troubleshooting

**Model not found?**
```bash
# Make sure the model is in the correct location
ls models/model.onnx
```

**Qdrant connection failed?**
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections
```

**Out of memory during chunking?**
```bash
# Reduce max_chunk_size or process fewer documents at once
let chunks = self.semantic_chunk_text(&doc, 0.7, 500)?;
```

---

Made with ❤️ and 🦀 Rust