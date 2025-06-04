import os
import shutil
from typing import List

import streamlit as st
from llama_index.core import (
    Settings,
    SummaryIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm = OpenAI(api_key=openai_api_key, temperature=0)

folder_path = "docs"

st.title("Ask My Docs 📄🧠")
st.write("Upload your documents and ask questions about them!")

uploaded_files = st.file_uploader(
    "Upload documents", accept_multiple_files=True, type=["pdf", "txt", "md"]
)


def vector_query(query: str, page_numbers: List[str]) -> str:
    """Perform a vector search over an index.

    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.

    """

    metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts, condition=FilterCondition.OR
        ),
    )
    response = query_engine.query(query)
    return response


if uploaded_files:
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.read())

    with st.spinner("Processing documents..."):
        reader = SimpleDirectoryReader(input_dir="docs")
        documents = reader.load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)
        vector_query_tool = FunctionTool.from_defaults(
            name="vector_tool", fn=vector_query
        )
        summary_index = SummaryIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        summary_tool = QueryEngineTool.from_defaults(
            name="summary_tool",
            query_engine=summary_query_engine,
            description=("Useful if you want to get a summary of the documents. "),
        )

    st.success("Documents processed! You can now ask questions.")

    question = st.text_input("Ask a question about your docs:")
    if question:
        with st.spinner("Thinking..."):
            response = llm.predict_and_call(
                [vector_query_tool, summary_tool], question, verbose=True
            )
            st.markdown(f"**Answer:** {response}")
