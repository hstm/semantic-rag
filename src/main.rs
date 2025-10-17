// Cargo.toml dependencies:
/*
[package]
name = "rag-ort"
version = "0.1.0"
edition = "2021"

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
tokenizers = "0.22.1"
*/

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use ort::{
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::Value,
};
use qdrant_client::{
    Qdrant,
    qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, PointStruct,
        UpsertPointsBuilder, VectorParams, VectorsConfig, SearchPointsBuilder,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use pdf_extract::extract_text_from_mem;
use regex::Regex;
use tower_http::cors::CorsLayer;
use tokenizers::{Tokenizer, PaddingParams, PaddingStrategy, TruncationParams};
use reqwest::Client;
use anyhow::Context;


// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone)]
struct AppState {
    rag: Arc<Mutex<RagSystem>>,
}

struct RagSystem {
    embedding_model: Session,
    qdrant_client: Qdrant,
    collection_name: String,
    vector_size: usize,
    tokenizer: Tokenizer,
    anthropic_client: Client,
    anthropic_api_key: String,
}

#[derive(Deserialize)]
struct AddDocumentsRequest {
    documents: Vec<String>,
}

#[derive(Deserialize)]
struct QueryRequest {
    question: String,
}

#[derive(Serialize)]
struct QueryResponse {
    answer: String,
    sources: Vec<Source>,
}

#[derive(Serialize)]
struct Source {
    text: String,
    distance: f32,
}

#[derive(Serialize)]
struct ApiResponse {
    success: bool,
    message: Option<String>,
    error: Option<String>,
}

// Anthropic API request/response structures
#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

// ============================================================================
// RAG System Implementation
// ============================================================================

impl RagSystem {
        async fn new(
        onnx_model_path: &str,
        tokenizer_path: &str,
        anthropic_api_key: String,
    ) -> anyhow::Result<Self> {
        // Initialize ONNX Runtime session
        println!("Loading ONNX model from: {}", onnx_model_path);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(onnx_model_path)?;

        println!("Model loaded successfully");

        // Initialize tokenizer
        println!("Loading tokenizer from: {}", tokenizer_path);
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Configure tokenizer with padding and truncation
        let max_length = 128;
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            stride: 0,
            ..Default::default()
        }));
        
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(max_length),
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));
        
        println!("Tokenizer loaded and configured successfully");

        // Connect to Qdrant (local instance)
        let qdrant_client = Qdrant::from_url("http://localhost:6333").build()?;
        
        let collection_name = "documents".to_string();
        // all-MiniLM-L6-v2 dimension
        // adjust based on the actual model used
        // cat models/config.json | grep hidden_size
        let vector_size = 384; 

        // Create collection if it doesn't exist
        let collections = qdrant_client.list_collections().await?;
        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if !collection_exists {
            println!("Creating collection: {}", collection_name);
            qdrant_client
                .create_collection(
                    CreateCollectionBuilder::new(&collection_name)
                        .vectors_config(VectorsConfig {
                            config: Some(Config::Params(VectorParams {
                                size: vector_size as u64,
                                distance: Distance::Cosine.into(),
                                ..Default::default()
                            })),
                        }),
                )
                .await?;
        } else {
            println!("Collection '{}' already exists", collection_name);
        }

        let anthropic_client = Client::new();
        
        Ok(Self {
            embedding_model: session,
            qdrant_client,
            collection_name,
            vector_size,
            tokenizer,
            anthropic_client,
            anthropic_api_key,
        })
    }

    fn tokenize(&self, text: &str) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
        // Tokenize the text
        let encoding = self.tokenizer
            .encode(text, true)  // true = add special tokens (CLS, SEP)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        
        // Get token IDs
        let tokens: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();
        
        // Get attention mask
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();
        
        Ok((tokens, attention_mask))
    }

    fn generate_embedding(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let (tokens, attention_mask) = self.tokenize(text)?;
        
        // Create input tensors - ort 2.0 wants (shape, data) tuple format
        let batch_size = 1;
        let seq_len = tokens.len();
        
        let tokens_value = Value::from_array((vec![batch_size, seq_len], tokens))?;
        let mask_value = Value::from_array((vec![batch_size, seq_len], attention_mask))?;
        
        // Run inference
        let outputs = self.embedding_model.run(ort::inputs![
            "input_ids" => tokens_value,
            "attention_mask" => mask_value,
        ])?;
        
        // Extract embeddings - try_extract_tensor returns the tensor directly
        let embeddings = outputs[0].try_extract_tensor::<f32>()?;
        
        // embeddings is already (&Shape, &[f32])
        let (_shape, data) = embeddings;
        let embedding_vec: Vec<f32> = data.iter().copied().collect();
        
        // Mean pooling (take first vector_size elements as sentence embedding)
        let sentence_embedding: Vec<f32> = embedding_vec
            .iter()
            .take(self.vector_size)
            .copied()
            .collect();
        
        // Normalize
        let norm: f32 = sentence_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = sentence_embedding.iter().map(|x| x / norm).collect();
        
        Ok(normalized)
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        // Simple sentence splitting using regex
        // Matches periods, exclamation marks, or question marks followed by whitespace or end of string
        let sentence_regex = Regex::new(r"(?<=[.!?])\s+(?=[A-Z])").unwrap();
        
        sentence_regex
            .split(text)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 10) // Filter out very short fragments
            .collect()
    }
    
    fn cosine_similarity(&self, vec_a: &[f32], vec_b: &[f32]) -> f32 {
        let dot_product: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = vec_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = vec_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    fn semantic_chunk_text(&mut self, text: &str, similarity_threshold: f32, max_chunk_size: usize) -> anyhow::Result<Vec<String>> {
        println!("  Splitting text into sentences...");
        let sentences = self.split_into_sentences(text);
        
        if sentences.is_empty() {
            return Ok(vec![]);
        }
        
        if sentences.len() == 1 {
            return Ok(sentences);
        }
        
        println!("  Found {} sentences, generating embeddings...", sentences.len());
        
        // Generate embeddings for all sentences
        let mut embeddings = Vec::new();
        for (i, sentence) in sentences.iter().enumerate() {
            let embedding = self.generate_embedding(sentence)?;
            embeddings.push(embedding);
            
            if (i + 1) % 10 == 0 {
                println!("    Embedded {}/{} sentences", i + 1, sentences.len());
            }
        }
        
        println!("  Grouping sentences by semantic similarity...");
        
        // Group sentences by similarity
        let mut chunks = Vec::new();
        let mut current_chunk = vec![sentences[0].clone()];
        let mut current_length = sentences[0].len();
        
        for i in 1..sentences.len() {
            // Calculate similarity with previous sentence
            let similarity = self.cosine_similarity(&embeddings[i - 1], &embeddings[i]);
            
            let next_length = current_length + sentences[i].len() + 1; // +1 for space
            
            // Decision criteria:
            // 1. Similar enough to previous sentence (semantic coherence)
            // 2. Won't exceed max chunk size
            let should_merge = similarity > similarity_threshold && next_length <= max_chunk_size;
            
            if should_merge {
                // Add to current chunk
                current_chunk.push(sentences[i].clone());
                current_length = next_length;
            } else {
                // Start new chunk
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.join(" "));
                }
                current_chunk = vec![sentences[i].clone()];
                current_length = sentences[i].len();
            }
            
            if (i + 1) % 20 == 0 {
                println!("    Processed {}/{} sentences into {} chunks so far", 
                         i + 1, sentences.len(), chunks.len());
            }
        }
        
        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.join(" "));
        }
        
        // Post-process: merge very small chunks with neighbors if possible
        let mut final_chunks = Vec::new();
        let mut i = 0;
        
        while i < chunks.len() {
            let chunk = &chunks[i];
            
            // If chunk is very small and not the last one, try to merge with next
            if chunk.len() < 200 && i + 1 < chunks.len() {
                let next_chunk = &chunks[i + 1];
                if chunk.len() + next_chunk.len() + 1 <= max_chunk_size {
                    final_chunks.push(format!("{} {}", chunk, next_chunk));
                    i += 2; // Skip next chunk since we merged it
                    continue;
                }
            }
            
            final_chunks.push(chunk.clone());
            i += 1;
        }
        
        println!("  Created {} semantic chunks (merged small chunks)", final_chunks.len());
        
        Ok(final_chunks)
    }
    
    // Simple chunking by paragraphs (not used now)
    // fn chunk_text(&mut self, text: &str, max_chunk_size: usize) -> anyhow::Result<Vec<String>> {
    //     let paragraphs: Vec<&str> = text.split("\n\n").collect();
    //     let mut chunks = Vec::new();
    //     let mut current_chunk = String::new();

    //     for para in paragraphs {
    //         let para = para.trim();
    //         if para.is_empty() {
    //             continue;
    //         }

    //         if current_chunk.len() + para.len() + 2 <= max_chunk_size {
    //             if !current_chunk.is_empty() {
    //                 current_chunk.push_str("\n\n");
    //             }
    //             current_chunk.push_str(para);
    //         } else {
    //             if !current_chunk.is_empty() {
    //                 chunks.push(current_chunk.clone());
    //             }
    //             current_chunk = para.to_string();
    //         }
    //     }

    //     if !current_chunk.is_empty() {
    //         chunks.push(current_chunk);
    //     }

    //     Ok(chunks)
    // }

async fn add_documents(&mut self, documents: Vec<String>) -> anyhow::Result<usize> {
    println!("Chunking {} documents...", documents.len());
    let mut all_chunks = Vec::new();
    
    for doc in documents {
        // Use semantic chunking instead of simple chunking
        let chunks = self.semantic_chunk_text(&doc, 0.7, 1000)?;
        all_chunks.extend(chunks);
    }
    
    println!("Created {} chunks", all_chunks.len());
    println!("Generating embeddings...");
    
    let mut points = Vec::new();
    for (i, chunk) in all_chunks.iter().enumerate() {
        let embedding = self.generate_embedding(chunk)?;  // Now chunk is &String which coerces to &str
        
        let point = PointStruct::new(
            i as u64,
            embedding,
            [("text".to_string(), chunk.clone().into())],
        );
        points.push(point);
        
        if (i + 1) % 10 == 0 {
            println!("Processed {}/{} chunks", i + 1, all_chunks.len());
        }
    }
    
    println!("Uploading to Qdrant...");
    self.qdrant_client
        .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
        .await?;
    
    Ok(all_chunks.len())
}

    async fn retrieve(&mut self, query: &str, top_k: usize) -> anyhow::Result<Vec<(String, f32)>> {
        let query_embedding = self.generate_embedding(query)?;
        
        let search_result = self.qdrant_client
            .search_points(
                SearchPointsBuilder::new(&self.collection_name, query_embedding, top_k as u64)
                    .with_payload(true),
            )
            .await?;
        
        let results: Vec<(String, f32)> = search_result
            .result
            .iter()
            .map(|point| {
                let text = point
                    .payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| String::new());
                let score = point.score;
                (text, score)
            })
            .collect();
        
        Ok(results)
    }

    async fn query(&mut self, question: &str) -> anyhow::Result<QueryResponse> {
        // Retrieve relevant sources
        let sources = self.retrieve(question, 5).await?;
        
        // Build context from sources
        let context: String = sources
            .iter()
            .enumerate()
            .map(|(i, (text, _))| format!("[Source {}]\n{}", i + 1, text))
            .collect::<Vec<_>>()
            .join("\n\n");
        
        // Create prompt for Claude
        let prompt = format!(
            "You are a helpful assistant answering questions based on the provided context. \
            Use the context below to answer the user's question. If the answer cannot be found \
            in the context, say so clearly.\n\n\
            Context:\n{}\n\n\
            Question: {}\n\n\
            Please provide a clear, concise answer based on the context above.",
            context, question
        );
        
        // Call Anthropic API
        let answer = self.call_anthropic_api(&prompt).await?;
        
        let sources_response: Vec<Source> = sources
            .iter()
            .map(|(text, distance)| Source {
                text: text.clone(),
                distance: *distance,
            })
            .collect();
        
        Ok(QueryResponse {
            answer,
            sources: sources_response,
        })
    }

    // Helper function to call Anthropic API
    async fn call_anthropic_api(&self, prompt: &str) -> anyhow::Result<String> {
        let request = AnthropicRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        };
        
        let response = self.anthropic_client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.anthropic_api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Anthropic API error: {}", error_text));
        }
        
        let api_response: AnthropicResponse = response.json().await?;
        
        let answer = api_response
            .content
            .first()
            .map(|block| block.text.clone())
            .ok_or_else(|| anyhow::anyhow!("No content in API response"))?;
        
        Ok(answer)
    }

    async fn clear_database(&self) -> anyhow::Result<()> {
        self.qdrant_client
            .delete_collection(&self.collection_name)
            .await?;
        
        self.qdrant_client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorsConfig {
                        config: Some(Config::Params(VectorParams {
                            size: self.vector_size as u64,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        })),
                    }),
            )
            .await?;
        
        Ok(())
    }
}

// ============================================================================
// HTTP Handlers
// ============================================================================

async fn index() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System with ONNX Runtime</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .input-section { margin-bottom: 20px; }
        .input-section label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        button {
            flex: 1;
            padding: 12px 24px;
            font-size: 1em;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); }
        .btn-secondary { background: #f0f0f0; color: #333; }
        .btn-danger { background: #e53e3e; color: white; }
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-weight: 600;
        }
        .answer-section {
            margin-top: 20px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .answer-section h3 { color: #667eea; margin-bottom: 10px; }
        .answer-text { line-height: 1.6; color: #333; }
        .sources { margin-top: 20px; }
        .sources h4 { color: #667eea; margin-bottom: 10px; }
        .source-item {
            background: white;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            border: 1px solid #e0e0e0;
            font-size: 0.9em;
        }
        .status {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab {
            padding: 10px 20px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
            font-size: 1em;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        .tab:hover {
            color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .file-upload {
            border: 2px dashed #667eea;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 15px;
            background: #f7fafc;
        }
        .file-upload:hover {
            background: #edf2f7;
            border-color: #5a67d8;
        }
        .file-upload-label {
            color: #667eea;
            font-weight: 600;
            font-size: 1.1em;
        }
        .file-name {
            margin-top: 15px;
            color: #666;
            font-size: 0.9em;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦀 RAG System</h1>
            <p>Powered by ONNX Runtime & Qdrant</p>
            <span class="badge">✨ True Semantic Chunking</span>
        </div>
        
        <div class="card">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('text')">📝 Text Input</button>
                <button class="tab" onclick="switchTab('pdf')">📄 PDF Upload</button>
            </div>
            
            <div id="text-tab" class="tab-content active">
                <div class="input-section">
                    <label>Add Documents:</label>
                    <textarea id="documents" rows="6" placeholder="Enter documents (separated by blank lines)"></textarea>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="addDocs()">Add Documents</button>
                        <button class="btn-danger" onclick="clearDB()">Clear Database</button>
                    </div>
                </div>
            </div>
            
            <div id="pdf-tab" class="tab-content">
                <div class="input-section">
                    <label>Upload PDF Files:</label>
                    <div class="file-upload" onclick="document.getElementById('pdfInput').click()">
                        <div class="file-upload-label">
                            📄 Click to select PDF files
                            <div style="font-size: 0.85em; margin-top: 8px; opacity: 0.7;">
                                You can select multiple files
                            </div>
                        </div>
                        <input type="file" id="pdfInput" accept=".pdf" multiple onchange="handleFileSelect(event)" style="display: none;">
                        <div id="fileName" class="file-name"></div>
                    </div>
                    <div class="button-group">
                        <button class="btn-secondary" onclick="uploadPDFs()">Upload & Process PDFs</button>
                    </div>
                </div>
            </div>
            
            <div id="status"></div>
            
            <div class="input-section">
                <label>Ask a Question:</label>
                <textarea id="question" rows="3" placeholder="What would you like to know?"></textarea>
                <div class="button-group">
                    <button class="btn-primary" onclick="askQuestion()">Ask Question</button>
                </div>
            </div>
            
            <div id="loading" style="display: none;" class="loading">Processing your query...</div>
            <div id="response"></div>
        </div>
    </div>
    
    <script>
        let selectedFiles = [];
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (tab === 'text') {
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('text-tab').classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('pdf-tab').classList.add('active');
            }
        }
        
        function handleFileSelect(event) {
            selectedFiles = Array.from(event.target.files);
            const fileNameDiv = document.getElementById('fileName');
            
            if (selectedFiles.length > 0) {
                const names = selectedFiles.map(f => f.name).join(', ');
                const totalSize = selectedFiles.reduce((sum, f) => sum + f.size, 0);
                const sizeMB = (totalSize / (1024 * 1024)).toFixed(2);
                fileNameDiv.innerHTML = `<strong>${selectedFiles.length} file(s) selected:</strong><br>${names}<br><span style="opacity: 0.7;">Total size: ${sizeMB} MB</span>`;
            } else {
                fileNameDiv.textContent = '';
            }
        }
        
        async function uploadPDFs() {
            const statusDiv = document.getElementById('status');
            
            if (selectedFiles.length === 0) {
                statusDiv.innerHTML = '<div class="status error">Please select PDF files first.</div>';
                return;
            }
            
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            statusDiv.innerHTML = '<div class="status">Uploading and processing PDFs...</div>';
            
            try {
                const res = await fetch('/upload_pdfs', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await res.json();
                
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                    selectedFiles = [];
                    document.getElementById('pdfInput').value = '';
                    document.getElementById('fileName').innerHTML = '';
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        async function addDocs() {
            const docs = document.getElementById('documents').value.split('\n\n').filter(d => d.trim());
            const statusDiv = document.getElementById('status');
            
            if (docs.length === 0) {
                statusDiv.innerHTML = '<div class="status error">Please enter some documents.</div>';
                return;
            }
            
            statusDiv.innerHTML = '<div class="status">Adding documents...</div>';
            
            try {
                const res = await fetch('/add_documents', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({documents: docs})
                });
                const data = await res.json();
                
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                    document.getElementById('documents').value = '';
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        async function clearDB() {
            if (!confirm('Clear all documents?')) return;
            const statusDiv = document.getElementById('status');
            
            try {
                const res = await fetch('/clear_database', {method: 'POST'});
                const data = await res.json();
                
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                } else {
                    statusDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                statusDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const responseDiv = document.getElementById('response');
            const loadingDiv = document.getElementById('loading');
            
            if (!question.trim()) {
                alert('Please enter a question.');
                return;
            }
            
            loadingDiv.style.display = 'block';
            responseDiv.innerHTML = '';
            
            try {
                const res = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const data = await res.json();
                loadingDiv.style.display = 'none';
                
                if (data.answer) {
                    let html = `
                        <div class="answer-section">
                            <h3>Answer:</h3>
                            <div class="answer-text">${data.answer.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;
                    
                    if (data.sources && data.sources.length > 0) {
                        html += '<div class="sources"><h4>Sources:</h4>';
                        data.sources.forEach((source, idx) => {
                            html += `<div class="source-item">
                                <strong>Source ${idx + 1} (Score: ${source.distance.toFixed(3)}):</strong><br>
                                ${source.text.substring(0, 200)}...
                            </div>`;
                        });
                        html += '</div>';
                    }
                    
                    responseDiv.innerHTML = html;
                } else {
                    responseDiv.innerHTML = `<div class="status error">Error: ${data.error}</div>`;
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                responseDiv.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
            }
        }
        
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
    "#)
}

async fn upload_pdfs_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    let mut documents = Vec::new();
    let mut file_count = 0;
    
    // Process each file in the multipart request
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: None,
                error: Some(format!("Failed to read multipart field: {}", e)),
            }),
        )
    })? {
        let _name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().unwrap_or("unknown").to_string();
        
        // Only process PDF files
        if file_name.to_lowercase().ends_with(".pdf") {
            let data = field.bytes().await.map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        success: false,
                        message: None,
                        error: Some(format!("Failed to read file data: {}", e)),
                    }),
                )
            })?;
            
            // Extract text from PDF
            match extract_text_from_mem(&data) {
                Ok(text) => {
                    println!("Extracted text from PDF: {} ({} bytes)", file_name, text.len());
                    documents.push(text);
                    file_count += 1;
                }
                Err(e) => {
                    println!("Warning: Failed to extract text from {}: {}", file_name, e);
                    // Continue processing other files
                }
            }
        }
    }
    
    if documents.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: None,
                error: Some("No valid PDF files found or extracted".to_string()),
            }),
        ));
    }
    
    // Add documents to RAG system
    let mut rag = state.rag.lock().await;
    match rag.add_documents(documents).await {
        Ok(chunk_count) => Ok(Json(ApiResponse {
            success: true,
            message: Some(format!(
                "Successfully processed {} PDF(s) and added {} chunks",
                file_count, chunk_count
            )),
            error: None,
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: None,
                error: Some(format!("Failed to add documents: {}", e)),
            }),
        )),
    }
}

async fn add_documents_handler(
    State(state): State<AppState>,
    Json(payload): Json<AddDocumentsRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    let mut rag = state.rag.lock().await;
    match rag.add_documents(payload.documents).await {
        Ok(count) => Ok(Json(ApiResponse {
            success: true,
            message: Some(format!("Successfully added {} chunks", count)),
            error: None,
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: None,
                error: Some(e.to_string()),
            }),
        )),
    }
}

async fn query_handler(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut rag = state.rag.lock().await;
    match rag.query(&payload.question).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            })),
        )),
    }
}

async fn clear_database_handler(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    let rag = state.rag.lock().await;
    match rag.clear_database().await {
        Ok(_) => Ok(Json(ApiResponse {
            success: true,
            message: Some("Database cleared successfully".to_string()),
            error: None,
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: None,
                error: Some(e.to_string()),
            }),
        )),
    }
}

// ============================================================================
// Main Application
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🦀 RAG System with ONNX Runtime + Claude");
    println!("==========================================\n");
    
    // Get API key from environment variable
    let anthropic_api_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY environment variable must be set")?;

    if anthropic_api_key.trim().is_empty() {
    anyhow::bail!("ANTHROPIC_API_KEY environment variable must not be empty");
    }
    
    // Initialize RAG system
    let mut rag = RagSystem::new(
        "models/model.onnx",
        "models/tokenizer.json",
        anthropic_api_key,
    ).await?;
    
    // Add example documents
    let example_docs = vec![
        r#"Artificial Intelligence (AI) is transforming healthcare in numerous ways. 
        Machine learning algorithms can now detect diseases from medical images with 
        accuracy rivaling human experts. AI-powered diagnostic tools analyze X-rays, 
        MRIs, and CT scans to identify conditions like cancer, pneumonia, and fractures.
    
        Natural language processing helps extract insights from medical records and 
        research papers. Predictive models forecast patient outcomes and identify 
        high-risk individuals who may benefit from early intervention."#.to_string(),

        r#"Climate change is causing significant impacts on global ecosystems. Rising 
        temperatures are leading to more frequent and severe weather events, including 
        hurricanes, droughts, and floods. The Arctic ice is melting at an alarming rate, 
        contributing to sea level rise that threatens coastal communities.
    
        Carbon emissions from fossil fuels are the primary driver of climate change. 
        Renewable energy sources like solar and wind power offer sustainable alternatives 
        that can help reduce greenhouse gas emissions and mitigate climate impacts."#.to_string(),

        r#"Quantum computing represents a paradigm shift in computational power. Unlike 
        classical computers that use bits (0 or 1), quantum computers use qubits that 
        can exist in multiple states simultaneously through superposition.
    
        This enables quantum computers to solve certain problems exponentially faster 
        than classical computers. Applications include cryptography, drug discovery, 
        optimization problems, and simulating quantum systems."#.to_string(),
    ];
    
    println!("Adding example documents...");
    rag.add_documents(example_docs).await?;
    println!("RAG system ready!\n");
    
    let state = AppState {
        rag: Arc::new(Mutex::new(rag)),
    };
    
    let app = Router::new()
        .route("/", get(index))
        .route("/add_documents", post(add_documents_handler))
        .route("/upload_pdfs", post(upload_pdfs_handler))
        .route("/query", post(query_handler))
        .route("/clear_database", post(clear_database_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);
    
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("🚀 Server running at http://127.0.0.1:3000\n");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}