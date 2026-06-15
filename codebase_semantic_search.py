import os
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

folder = "project_folder"

all_functions = []

# Scan all Python files
for file in os.listdir(folder):

    filepath = os.path.join(folder, file)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):

            docstring = ast.get_docstring(node) or "No docstring"

            chunk = f"""
File:
{file}

Function Name:
{node.name}

Docstring:
{docstring}

Code:
{ast.get_source_segment(source, node)}
"""

            all_functions.append({
                "file": file,
                "name": node.name,
                "doc": docstring,
                "code": ast.get_source_segment(source, node),
                "embedding": model.encode(chunk)
            })

# User query
query = input("Enter search query: ")

query_embedding = model.encode([query])

results = []

# Compare query against every function
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

# Rank results
ranked_results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

print("\nTop Results:\n")

for i, result in enumerate(ranked_results[:3], start=1):

    print(f"{i}. {result['name']}")
    print(f"   File: {result['file']}")
    print(f"   Score: {result['score']:.4f}")

    print("\n   Code:")
    print(result["code"])

    print("-" * 50)