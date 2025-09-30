from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

cats = ['alt.atheism','comp.graphics','sci.space','rec.sport.baseball']
train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))
#test  = fetch_20newsgroups(subset='test',  categories=cats, remove=('headers','footers','quotes'))

docs = train.data[:400]   # use a subset
cv = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
tf = cv.fit_transform(docs)

n_topics = 6
lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method='batch', max_iter=10)
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print("Topic %d:" % topic_idx, " ".join(top_features))

print_top_words(lda, cv.get_feature_names_out(), 12)