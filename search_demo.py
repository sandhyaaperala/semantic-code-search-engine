from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "user authentication system",
    "database connection",
    "payment gateway integration",
    "JWT token verification"
]

query = "verify access token"

doc_embeddings = model.encode(documents)

query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)
scores_list = scores[0]

ranked_results = sorted(
    zip(documents, scores_list),
    key=lambda x: x[1],
    reverse=True
)

print("Top Results:\n")

for doc, score in ranked_results:
    print(f"{doc} → {score:.4f}")