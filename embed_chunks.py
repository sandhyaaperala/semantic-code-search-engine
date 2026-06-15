from sentence_transformers import SentenceTransformer
import ast

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_functions(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):

            results.append({
                "name": node.name,
                "doc": ast.get_docstring(node) or "No docstring",
                "code": ast.get_source_segment(source, node)
            })

    return results


functions = extract_functions("sample_code.py")

for fn in functions:

    chunk = f"""
Function Name:
{fn['name']}

Docstring:
{fn['doc']}

Code:
{fn['code']}
"""

    embedding = model.encode(chunk)

    print("=" * 50)
    print("Function:", fn["name"])
    print("Embedding Length:", len(embedding))