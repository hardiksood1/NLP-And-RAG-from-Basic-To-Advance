# 5. Evaluation Metrics.ipynb
# Deep Learning for NLP

# ---------------------------
# 1️⃣ Classification Metrics
# ---------------------------

# Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Sample classification labels
y_true = [0, 1, 0, 1, 1, 0]  # ground truth
y_pred = [0, 1, 0, 0, 1, 1]  # model predictions

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision, Recall, F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"])
print("Classification Report:\n", report)

# ---------------------------
# 2️⃣ Sequence Labeling Metrics (NER Example)
# ---------------------------

# Install seqeval if not installed
# !pip install seqeval

from seqeval.metrics import classification_report as seq_classification_report, f1_score as seq_f1_score

# Sample NER tags
y_true_ner = [["O", "B-PER", "I-PER", "O"]]
y_pred_ner = [["O", "B-PER", "O", "O"]]

print("NER F1-score:", seq_f1_score(y_true_ner, y_pred_ner))
print("NER Classification Report:\n", seq_classification_report(y_true_ner, y_pred_ner))

# ---------------------------
# 3️⃣ Text Generation Metrics
# ---------------------------

# BLEU Score Example
from nltk.translate.bleu_score import sentence_bleu

reference = [["this", "is", "a", "test"]]
candidate = ["this", "is", "test"]

bleu_score = sentence_bleu(reference, candidate)
print("BLEU Score:", bleu_score)

# ROUGE Score Example
# !pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score("The cat sat on the mat", "The cat is on the mat")
print("ROUGE Scores:", scores)

# Perplexity Example
import math

# Example: assume log-likelihood of a sentence
log_likelihood = -10.5
N = 5  # number of words
perplexity = math.exp(-log_likelihood/N)
print("Perplexity:", perplexity)

# ---------------------------
# 4️⃣ Regression Metrics (Optional for NLP tasks)
# ---------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sample regression values
y_true_reg = [3.0, 2.5, 4.0]
y_pred_reg = [2.8, 2.7, 3.9]

mse = mean_squared_error(y_true_reg, y_pred_reg)
mae = mean_absolute_error(y_true_reg, y_pred_reg)
r2 = r2_score(y_true_reg, y_pred_reg)

print("Regression Metrics:")
print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)

# ---------------------------
# ✅ Summary Table
# ---------------------------

print("""
Summary of Metrics:
- Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Sequence Labeling (NER): Entity-level F1, Precision, Recall
- Text Generation: BLEU, ROUGE, Perplexity
- Regression (optional): MSE, MAE, R2
""")



# Output 

# Accuracy: 0.6666666666666666
# Precision: 0.6666666666666666
# Recall: 0.6666666666666666
# F1-score: 0.6666666666666666
# Classification Report:
#                precision    recall  f1-score   support

#      Class 0       0.67      0.67      0.67         3
#      Class 1       0.67      0.67      0.67         3

#     accuracy                           0.67         6
#    macro avg       0.67      0.67      0.67         6
# weighted avg       0.67      0.67      0.67         6

# NER F1-score: 0.0
# NER Classification Report:
#                precision    recall  f1-score   support

#          PER       0.00      0.00      0.00         1

#    micro avg       0.00      0.00      0.00         1
#    macro avg       0.00      0.00      0.00         1
# weighted avg       0.00      0.00      0.00         1

# C:\Users\Hsood\AppData\Local\Programs\Python\Python310\lib\site-packages\nltk\translate\bleu_score.py:577: UserWarning: 
# The hypothesis contains 0 counts of 3-gram overlaps.
# Therefore the BLEU score evaluates to 0, independently of
# how many N-gram overlaps of lower order it contains.
# Consider using lower n-gram order or use SmoothingFunction()
#   warnings.warn(_msg)
# C:\Users\Hsood\AppData\Local\Programs\Python\Python310\lib\site-packages\nltk\translate\bleu_score.py:577: UserWarning: 
# The hypothesis contains 0 counts of 4-gram overlaps.
# Therefore the BLEU score evaluates to 0, independently of
# how many N-gram overlaps of lower order it contains.
# Consider using lower n-gram order or use SmoothingFunction()
#   warnings.warn(_msg)
# BLEU Score: 8.987727354491445e-155
# ROUGE Scores: {'rouge1': Score(precision=0.8333333333333334, recall=0.8333333333333334, fmeasure=0.8333333333333334), 'rougeL': Score(precision=0.8333333333333334, recall=0.8333333333333334, fmeasure=0.8333333333333334)}
# Perplexity: 8.16616991256765
# Regression Metrics:
# MSE: 0.030000000000000054
# MAE: 0.16666666666666682
# R2 Score: 0.9228571428571427

# Summary of Metrics:
# - Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
# - Sequence Labeling (NER): Entity-level F1, Precision, Recall
# - Text Generation: BLEU, ROUGE, Perplexity
# - Regression (optional): MSE, MAE, R2