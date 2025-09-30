from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
documents = ["I love AI", "AI is the future", "I love learning"]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("PCA Reduced Data:\n", X_reduced)

#Result
#PCA Reduced Data:
# [[ 0.13759916  0.57544887]
# [-0.77082821 -0.2031325 ]
# [ 0.63322906 -0.37231637]]