from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = [
    "I love AI and machine learning",
    "AI is the future",
    "Deep learning is a part of AI",
    "Artificial Intelligence will change the world",

    "I hate bugs and crashes",
    "Bugs make me angry",
    "Fixing errors is frustrating",
    "Crashes ruin the experience",

    "Space exploration is amazing",
    "NASA explores space",
    "Astronauts train hard for space missions",
    "The universe is vast and mysterious"
]

# 0=negative-bugs, 1=ai-positive, 2=space
y = [1,1,1,1, 0,0,0,0, 2,2,2,2]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.33, random_state=42, stratify=y
)

def train_and_report(clf_pipeline, name):
    clf_pipeline.fit(X_train, y_train)
    preds = clf_pipeline.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, preds, zero_division=0))  # <-- suppress warning

# Naive Bayes
mnb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('clf', MultinomialNB())
])
train_and_report(mnb, "MultinomialNB")

# Logistic Regression
logr = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))
])
train_and_report(logr, "LogisticRegression")

# Linear SVC
svc = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('clf', LinearSVC(max_iter=2000))
])
train_and_report(svc, "LinearSVC")




#output

# === MultinomialNB ===
#               precision    recall  f1-score   support

#            0       0.50      1.00      0.67         1
#            1       0.00      0.00      0.00         1
#            2       1.00      1.00      1.00         2

#     accuracy                           0.75         4
#    macro avg       0.50      0.67      0.56         4
# weighted avg       0.62      0.75      0.67         4

# C:\Users\Hsood\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\linear_model\_logistic.py:1272: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.8. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
#   warnings.warn(
# === LogisticRegression ===
#               precision    recall  f1-score   support

#            0       0.25      1.00      0.40         1
#            1       0.00      0.00      0.00         1
#            2       0.00      0.00      0.00         2

#     accuracy                           0.25         4
#    macro avg       0.08      0.33      0.13         4
# weighted avg       0.06      0.25      0.10         4

# === LinearSVC ===
#               precision    recall  f1-score   support

#            0       0.50      1.00      0.67         1
#            1       0.00      0.00      0.00         1
#            2       1.00      1.00      1.00         2

#     accuracy                           0.75         4
#    macro avg       0.50      0.67      0.56         4
# weighted avg       0.62      0.75      0.67         4