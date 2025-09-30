# Feature Engineering & Classical Models

This repository contains small, self-contained projects showcasing core concepts in **Natural Language Processing (NLP)** using Python and popular machine learning libraries.  

---

##  Project Contents

### 1. Text Classification with Classical ML
**File:** `1.Text classification with classical ML (Naïve Bayes, Logistic Regression, SVM).py`  
- Implements supervised classification on small text samples.  
- Models used:
  - Naïve Bayes
  - Logistic Regression
  - Linear SVM  
- Uses **TF-IDF** with n-grams for feature extraction.  
- Outputs precision, recall, and F1-score reports.  

---

### 2. N-grams & Simple Language Models
**File:** `2.n-grams & simple language models.py`  
- Demonstrates construction of **unigrams** and **bigrams**.  
- Computes conditional probabilities like *P(next word | current word)*.  
- Example: Probability of seeing *"ai"* after *"love"* in a toy corpus.  

---

### 3. Topic Modeling with LDA
**File:** `3.Topic modeling — LDA (Latent Dirichlet Allocation).py`  
- Applies **Latent Dirichlet Allocation (LDA)** for topic modeling.  
- Dataset: [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).  
- Extracts key words per topic to interpret hidden structures.  

---
python "2.n-grams & simple language models.py"
python "3.Topic modeling — LDA (Latent Dirichlet Allocation).py"
