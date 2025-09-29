from gensim.models import Word2Vec

# Tokenized text corpus
sentences = [
    ["i", "love", "nlp"],
    ["nlp", "is", "fun"],
    ["i", "love", "machine", "learning"]
]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# Word vector
print("Vector for 'nlp':\n", model.wv['nlp'])

# Similar words
print("Most similar to 'nlp':", model.wv.most_similar("nlp"))