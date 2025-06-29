import os
import time
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

# 🔑 Load API Key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# 🎛️ Set up embedding model and LLM
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(api_key=openai_api_key, temperature=0, model="gpt-4.1")
Settings.llm = llm

# 🎨 Streamlit UI
st.title("Ask My Docs 📄🧠")
st.write("Upload your documents and ask questions — now with **multi-hop reasoning**!")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt", "md"])

if uploaded_files:
    os.makedirs("docs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.read())

    with st.spinner("Processing documents..."):
        reader = SimpleDirectoryReader(input_dir="docs")
        documents = reader.load_data()

        splitter = SentenceSplitter(chunk_size=256)
        nodes = splitter.get_nodes_from_documents(documents)

        # Create indexes
        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

        # Create query engines
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
        vector_query_engine = vector_index.as_query_engine(similarity_top_k=10)
        bm25_query_engine = RetrieverQueryEngine.from_args(retriever=bm25_retriever)

        # Create tool wrappers
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for summarizing the full document set.",
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Best for conceptual and semantic similarity-based retrieval.",
        )

        bm25_tool = QueryEngineTool.from_defaults(
            query_engine=bm25_query_engine,
            description="Best for keyword-based, exact text retrieval.",
        )

        # Set up a router query engine (optional)
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                vector_tool,
                # summary_tool,  # Enable if needed
                # bm25_tool,     # Enable if needed
            ],
            verbose=True
        )

    st.success("✅ Documents loaded! Ready to ask questions with multi-hop reasoning.")

    # 💡 Multi-hop reasoning logic
    def multi_hop_query(original_question, retriever_query_engine, llm, max_hops=5):
        context_history = []
        for hop in range(max_hops):
            st.info(f"🔁 **Step {hop+1}: Asking LLM for next sub-question...**")

            subq_prompt = f"""
You are helping answer the question: "{original_question}"

So far, you have the following information:
"{' '.join(context_history) if context_history else '[No prior context]'}"

What should be the next sub-question you need to answer?
Only return the sub-question, no explanation.
"""
            subq_response = llm.complete(subq_prompt)
            next_sub_question = subq_response.text.strip()
            st.markdown(f"**Sub-question {hop+1}:** {next_sub_question}")

            st.info(f"🔍 **Retrieving documents for sub-question...**")
            retrieved_response = retriever_query_engine.query(next_sub_question)
            retrieved_text = str(retrieved_response)
            context_history.append(retrieved_text)

            st.success(f"✅ Retrieved context added.")
            time.sleep(0.5)

            stop_prompt = f"""
Original question: "{original_question}"
Context so far: "{' '.join(context_history)}"

Do you now have enough information to answer the question?
Respond with "yes" or "no" only.
"""
            stop_check = llm.complete(stop_prompt).text.lower().strip()
            if "yes" in stop_check:
                st.info("🛑 LLM determined it has enough information to answer.")
                break

        final_prompt = f"""
Answer the original question based on the context below.

Question: "{original_question}"

Context: "{' '.join(context_history)}"

Answer:"""

        final_answer = llm.complete(final_prompt).text.strip()
        return final_answer

    # 🧠 Main QA interface
    question = st.text_input("Ask a question about your docs (multi-hop supported):")
    if question:
        with st.spinner("🤔 Reasoning across multiple steps..."):
            answer = multi_hop_query(
                original_question=question,
                retriever_query_engine=vector_query_engine,
                llm=llm,
                max_hops=5
            )
            st.markdown(f"### 🧠 Final Answer:\n{answer}")
