from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["I love NLP", "NLP is fun", "I love Machine Learning"]

# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)

print("Vocabulary:", tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", X.toarray())