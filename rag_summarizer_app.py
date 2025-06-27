import streamlit as st
from RAGSummarizer import RAGSummarizer
import tempfile
import os

# Initialize the RAGSummarizer once
summarizer = RAGSummarizer()

# Page configuration
st.set_page_config(
    page_title="RAG Document Summarizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        color: #1E88E5;
        font-weight: bold;
    }
    .sub-header {
        font-size: 20px;
        color: #424242;
        margin-bottom: 20px;
    }
    .summary-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f7ff;
        border-left: 5px solid #1E88E5;
    }
    .footer {
        text-align: center;
        color: #616161;
        font-size: 14px;
        margin-top: 50px;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
    }
    .made-by {
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section
st.markdown('<p class="main-header">📄 RAG Document Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a document and get an intelligent summary using Retrieval-Augmented Generation (RAG)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2665/2665038.png", width=100)
    st.markdown("### About")
    st.markdown("""
    This tool uses RAG technology to create contextually relevant document summaries.
    
    Supported file formats:
    - PDF
    - TXT
    - Markdown
    """)
    
    st.markdown("### How to use")
    st.markdown("""
    1. Upload your document
    2. Enter your query or use the default
    3. Click 'Generate Summary'
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # File uploader
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "md"])

with col2:
    # Query input
    query = st.text_input("Enter your query", value="Summarize this document")

# Submit button with better styling
if uploaded_file:
    # Check if file is not empty
    if uploaded_file.size == 0:
        st.error("The uploaded file is empty. Please upload a valid document.")
    else:
        generate_button = st.button("✨ Generate Summary", type="primary", use_container_width=True)
        
        if generate_button:
            with st.spinner("🔍 Processing document..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    result = summarizer.process_document(tmp_path, query=query)
                    
                    # Validate result structure
                    required_keys = ["summary", "retrieved_context", "latency", "similarity_scores"]
                    if not all(key in result for key in required_keys):
                        missing_keys = [key for key in required_keys if key not in result]
                        raise KeyError(f"Missing keys in result: {', '.join(missing_keys)}")
                    
                    # Create tabs for better organization
                    summary_tab, context_tab, metrics_tab = st.tabs(["📝 Summary", "📚 Retrieved Context", "📊 Metrics"])
                    
                    with summary_tab:
                        st.markdown(f"### Summary for: {uploaded_file.name}")
                        st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
                    
                    with context_tab:
                        for i, chunk in enumerate(result["retrieved_context"], 1):
                            with st.expander(f"Chunk {i}"):
                                st.text(chunk)

                    with metrics_tab:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processing Time (s)", f"{round(result['latency'], 2)}")
                        with col2:
                            st.metric("Chunks Retrieved", len(result["retrieved_context"]))
                        
                        st.subheader("Similarity Scores")
                        st.json(result["similarity_scores"])

                except KeyError as e:
                    st.error(f"Error in result format: {e}")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

                finally:
                    os.remove(tmp_path)  # Clean up
else:
    st.info("Please upload a document to get started")

# Footer with attribution
st.markdown("""
<div class="footer">
    <p>Made by <span class="made-by">Saim Nadeem</span> | © 2025</p>
</div>
""", unsafe_allow_html=True)