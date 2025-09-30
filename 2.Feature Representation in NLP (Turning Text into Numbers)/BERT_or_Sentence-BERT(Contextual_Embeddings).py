from sentence_transformers import SentenceTransformer

# Load pretrained BERT-based model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences
sentences = ["I love NLP", "NLP is fun", "I love Machine Learning"]

# Generate embeddings
embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("Embedding for first sentence:\n", embeddings[0])