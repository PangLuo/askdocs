import os
import streamlit as st
from llama_index.core import Settings, SummaryIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm = OpenAI(api_key=openai_api_key, temperature=0)

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
        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = vector_index.as_query_engine()
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                "Useful for summarization questions"
            ),
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context"
            ),
        )
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool,
            ],
            verbose=True
        )

    st.success("Documents processed! You can now ask questions.")

    question = st.text_input("Ask a question about your docs:")
    if question:
        with st.spinner("Thinking..."):
            response = query_engine.query(question)
            st.markdown(f"**Answer:** {response}")
