from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt

# Larger dataset (10 samples instead of 3)
documents = [
    "I love AI", "AI is the future", "I love learning",
    "Machine learning is amazing", "Deep learning with Python",
    "Natural language processing is fun", "Data science is powerful",
    "I enjoy coding", "AI will change the world", "Python is great"
]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Apply UMAP
umap_model = umap.UMAP(n_components=2, n_neighbors=3, random_state=42)
X_umap = umap_model.fit_transform(X)

# Print reduced data
print("UMAP Reduced Data:\n", X_umap)

# Plot
plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("UMAP Visualization")
plt.show()

#Result
# UMAP Reduced #Data:
# [[15.7315235 13.784465 ]
# [16.303299  14.785437 ]
# [15.828559  12.979677 ]
# [16.350058  12.940301 ]
# [17.312859  14.131574 ]
# [17.207417  16.07606  ]
# [17.785194  15.159201 ]
# [16.64579   15.991948 ]
# [15.764555  14.572206 ]
# [17.296722  15.038333 ]]
