import ast

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
                "line": node.lineno,
                "code": ast.get_source_segment(source, node)
            })

    return results


# Testing with our sample code file
for fn in extract_functions("sample_code.py"):
    print("=" * 50)
    print("Function :", fn["name"])
    print("Line     :", fn["line"])
    print("Docstring:", fn["doc"])
    print("\nCode:")
    print(fn["code"])
    print()