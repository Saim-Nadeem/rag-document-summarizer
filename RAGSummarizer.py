import os
import PyPDF2
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer
from typing import List, Tuple
import time
import logging

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class RAGSummarizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", llm_model: str = "facebook/bart-large-cnn"):
        """Initialize the RAG summarizer with embedding and LLM models."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embedder = SentenceTransformer(model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.llm = pipeline("summarization", model=llm_model, device=-1, max_length=1024)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

    def ingest_document(self, file_path: str) -> List[str]:
        """Ingest and split document into chunks."""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up excessive newlines and spaces
                        page_text = re.sub(r'\n\s*\n', '\n\n', page_text)  # Normalize multiple newlines
                        page_text = re.sub(r'\s+', ' ', page_text)  # Normalize multiple spaces
                        text += page_text + "\n"
        elif file_path.endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r'\s+', ' ', text)
        else:
            raise ValueError("Unsupported file format. Use PDF, TXT, or Markdown.")

        if not text.strip():
            raise ValueError("Document is empty or could not be read.")

        print(f"Debug: Raw extracted text:\n{text[:500]}...")  # Show first 500 chars
        self.chunks = self.text_splitter.split_text(text)
        print(f"Debug: Number of chunks created: {len(self.chunks)}")
        print(f"Debug: First chunk sample:\n{self.chunks[0][:200] if self.chunks else 'No chunks'}...")
        return self.chunks

    def create_embeddings(self) -> None:
        """Create and store embeddings for document chunks in FAISS."""
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        self.index.reset()
        self.index.add(embeddings)
        print(f"Debug: Embeddings shape: {embeddings.shape}")

    def retrieve_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant chunks for the query."""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        top_k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][i])))
        print(f"Debug: Retrieved {len(results)} chunks with distances: {distances[0][:len(results)]}")
        return results

    def generate_summary(self, retrieved_chunks: List[Tuple[str, float]], max_length: int = 200) -> dict:
        """Generate summary from retrieved chunks."""
        start_time = time.time()
        if not retrieved_chunks:
            return {
                "summary": "Error: No relevant chunks retrieved.",
                "retrieved_context": [],
                "similarity_scores": [],
                "latency": time.time() - start_time
            }

        context = " ".join([chunk.strip() for chunk, _ in retrieved_chunks if chunk.strip()])
        print(f"Debug: Context length before truncation: {len(context)} characters")
        tokens = self.tokenizer.encode(context, truncation=True, max_length=1000)
        truncated_context = self.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Debug: Truncated context length: {len(truncated_context)} characters")

        try:
            summary = self.llm(truncated_context, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"Error during summarization: {str(e)}"

        latency = time.time() - start_time
        return {
            "summary": summary,
            "retrieved_context": [chunk for chunk, _ in retrieved_chunks],
            "similarity_scores": [score for _, score in retrieved_chunks],
            "latency": latency
        }

    def process_document(self, file_path: str, query: str = "Summarize this document") -> dict:
        """Process document and generate summary."""
        self.chunks = self.ingest_document(file_path)
        self.create_embeddings()
        retrieved_chunks = self.retrieve_chunks(query)
        result = self.generate_summary(retrieved_chunks)
        return result

def main():
    summarizer = RAGSummarizer()
    sample_pdf = "doc1.pdf"
    if not os.path.exists(sample_pdf):
        print("Sample document not found. Please provide a valid PDF, TXT, or Markdown file.")
        return
    
    result = summarizer.process_document(sample_pdf)
    
    print("\n=== Summary ===")
    print(result["summary"])
    print("\n=== Retrieved Context ===")
    for i, chunk in enumerate(result["retrieved_context"], 1):
        print(f"Chunk {i}:\n{chunk}\n")
    print("=== Metadata ===")
    print(f"Similarity Scores: {result['similarity_scores']}")
    print(f"Latency: {result['latency']:.2f} seconds")

if __name__ == "__main__":
    main()
