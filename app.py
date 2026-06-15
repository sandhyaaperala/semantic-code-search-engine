import json
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Semantic Code Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Code Search Engine")

st.write(
    "Search your codebase using natural language queries. "
    "Powered by Sentence Transformers and AST parsing."
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load indexed functions
with open("index.json", "r", encoding="utf-8") as f:
    all_functions = json.load(f)

query = st.text_input(
    "Enter your search query",
    placeholder="Example: verify login credentials"
)

if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a search query.")
        st.stop()

    query_embedding = model.encode([query])

    results = []

    for fn in all_functions:

        score = cosine_similarity(
            query_embedding,
            [fn["embedding"]]
        )

        results.append({
            "file": fn["file"],
            "name": fn["name"],
            "code": fn["code"],
            "score": score[0][0]
        })

    ranked_results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    st.subheader("Top Results")

    for i, result in enumerate(ranked_results[:3], start=1):

        st.markdown(f"## {i}. {result['name']}")

        st.write(f"**File:** `{result['file']}`")
        st.write(f"**Similarity Score:** `{result['score']:.4f}`")

        st.code(
            result["code"],
            language="python"
        )

        st.divider()