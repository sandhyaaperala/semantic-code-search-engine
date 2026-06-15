import os
import ast
import json
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

            embedding = model.encode(chunk)

            all_functions.append({
                "file": file,
                "name": node.name,
                "doc": docstring,
                "code": ast.get_source_segment(source, node),
                "embedding": embedding.tolist()
            })

with open("index.json", "w", encoding="utf-8") as f:
    json.dump(all_functions, f, indent=4)

print("Index created successfully!")
print("Functions indexed:", len(all_functions))