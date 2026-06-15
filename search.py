import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index
with open("index.json", "r", encoding="utf-8") as f:
    all_functions = json.load(f)

# User query
query = input("Enter search query: ")

query_embedding = model.encode([query])

results = []

# Calculate similarity scores
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

# Sort results
ranked_results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

# Debug: show all scores
print("\nAll Scores:")
for r in ranked_results:
    print(f"{r['name']} -> {r['score']:.4f}")

# Threshold filtering
THRESHOLD = 0.20

filtered_results = [
    r for r in ranked_results
    if r["score"] >= THRESHOLD
]

# No matches found
if not filtered_results:
    print("\nNo relevant functions found.")
    exit()

print(f"\nFound {len(filtered_results)} relevant matches\n")

# Show top 3 results
for i, result in enumerate(filtered_results[:3], start=1):

    print(f"{i}. {result['name']}")
    print(f"   File: {result['file']}")
    print(f"   Score: {result['score']:.4f}")

    print("\n   Code:")
    print(result["code"])

    print("-" * 50)