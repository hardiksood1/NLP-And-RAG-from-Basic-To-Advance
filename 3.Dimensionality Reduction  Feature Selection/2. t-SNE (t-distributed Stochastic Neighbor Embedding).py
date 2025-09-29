from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample text data
documents = ["I love AI", "AI is the future", "I love learning"]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Apply t-SNE with smaller perplexity
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Print reduced data
print("t-SNE Reduced Data:\n", X_tsne)

# Plot the reduced data
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.show()

#Result
#t-SNE Reduced Data:
# [[-223.24127   199.56741 ]
# [ -80.42361   -29.748756]
# [  46.500675  209.49757 ]]