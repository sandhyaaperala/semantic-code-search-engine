import os
import ast
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

folder = "project_folder"

all_functions = []

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

print(f"\nTotal Functions Found: {len(all_functions)}\n")

for fn in all_functions:
    print(fn["file"], "→", fn["name"])
    print("Embedding Length:", len(fn["embedding"]))
    print()