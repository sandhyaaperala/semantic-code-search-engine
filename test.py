from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode("authentication")

print(embedding[:10])
print("Length:", len(embedding))