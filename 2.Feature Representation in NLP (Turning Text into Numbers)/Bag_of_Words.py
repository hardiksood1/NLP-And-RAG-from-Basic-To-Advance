from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
docs = ["I love NLP", "NLP is fun", "I love Machine Learning"]

# Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", X.toarray())