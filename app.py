import os
import streamlit as st
from llama_index.core import Settings, SummaryIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set LLM
llm = OpenAI(api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo")
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

st.title("Ask My Docs 📄🧠")
st.write("Upload your documents and ask questions about them!")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt", "md"])

if uploaded_files:
    os.makedirs("docs", exist_ok=True)
    
    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.read())

    with st.spinner("Processing documents..."):
        reader = SimpleDirectoryReader(input_dir="docs")
        documents = reader.load_data()
        
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        # Create indexes
        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)
        
        # Create BM25 retriever for full-text search
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=3
        )

        # Create query engines
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        
        vector_query_engine = vector_index.as_query_engine()
        
        # Create BM25-based query engine
        bm25_query_engine = RetrieverQueryEngine.from_args(
            retriever=bm25_retriever,
            response_synthesizer=None  # Uses default response synthesizer
        )

        # Create tools
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                "Useful for summarization questions related to the entire document collection. "
                "Best for questions asking for overviews, general summaries, or broad insights."
            ),
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context using semantic similarity. "
                "Best for conceptual questions and finding semantically related information."
            ),
        )
        
        bm25_tool = QueryEngineTool.from_defaults(
            query_engine=bm25_query_engine,
            description=(
                "Useful for keyword-based full-text search using BM25 ranking. "
                "Best for finding exact terms, phrases, or specific factual information "
                "when you know the keywords that should appear in the answer."
            ),
        )

        # Create router query engine with all three tools
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool,
                bm25_tool,
            ],
            verbose=True
        )

    st.success("Documents processed! You can now ask questions using three different search methods:")
    st.info("""
    **Available Search Methods:**
    - 📋 **Summary Tool**: For document overviews and general summaries
    - 🔍 **Vector Search**: For semantic/conceptual questions  
    - 🎯 **BM25 Full-text**: For exact keyword and phrase matching
    
    The system will automatically choose the best method for your question!
    """)

    question = st.text_input("Ask a question about your docs:")
    
    if question:
        with st.spinner("Thinking..."):
            response = query_engine.query(question)
            st.markdown(f"**Answer:** {response}")
            
            # Optional: Show which tool was selected (if verbose mode provides this info)
            if hasattr(response, 'metadata') and 'selector_result' in response.metadata:
                selected_tool = response.metadata['selector_result']
                st.caption(f"*Search method used: {selected_tool}*")