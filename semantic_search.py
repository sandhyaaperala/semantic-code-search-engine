from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_functions(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):

            chunk = f"""
Function Name:
{node.name}

Docstring:
{ast.get_docstring(node) or "No docstring"}

Code:
{ast.get_source_segment(source, node)}
"""

            results.append({
                "name": node.name,
                "code": ast.get_source_segment(source, node),
                "embedding": model.encode(chunk)
            })

    return results


functions = extract_functions("sample_code.py")

query = "create database connection"

query_embedding = model.encode([query])

results = []

for fn in functions:

    score = cosine_similarity(
        query_embedding,
        [fn["embedding"]]
    )

    results.append({
        "name": fn["name"],
        "code": fn["code"],
        "score": score[0][0]
    })

ranked_results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

best_match = ranked_results[0]

print("\nBest Match:\n")

print("Function:")
print(best_match["name"])

print("\nScore:")
print(best_match["score"])

print("\nCode:")
print(best_match["code"])