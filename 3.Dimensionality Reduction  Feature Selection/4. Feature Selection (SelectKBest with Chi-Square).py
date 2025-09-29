from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

# Sample documents
documents = ["I love AI", "AI is the future", "I love learning"]
y = [1, 0, 1]  # Example labels

# Convert text to count vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Apply Chi-Square Feature Selection
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)

# Print results
print("Selected Features:\n", X_new.toarray())
selected_feature_names = [vectorizer.get_feature_names_out()[i] for i in selector.get_support(indices=True)]
print("Feature Names:", selected_feature_names)

# -------- Visualization --------

# Chi2 scores for all features
feature_scores = selector.scores_

# All feature names
all_features = vectorizer.get_feature_names_out()

# Plot feature importance
plt.figure(figsize=(8, 4))
plt.bar(all_features, feature_scores, color="skyblue")
plt.title("Chi-Square Scores for Features")
plt.xlabel("Features (Words)")
plt.ylabel("Chi2 Score")

# Highlight selected features
for idx, name in enumerate(all_features):
    if name in selected_feature_names:
        plt.bar(name, feature_scores[idx], color="orange")

plt.show()


#Result
# Selected Features:
# [[0 0]
# [1 1]
# [0 0]]
# Feature Names: ['is', 'the']
