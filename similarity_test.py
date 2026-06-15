from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "authentication"
text2 = "pizza"

embedding1 = model.encode([text1])
embedding2 = model.encode([text2])

score = cosine_similarity(embedding1, embedding2)

print(score)