#  Deep Learning for NLP 

This repository contains a complete step-by-step learning series on **Deep Learning for Natural Language Processing (NLP)**.  
Each script demonstrates a major concept — from simple neural networks to attention-based Seq2Seq models and evaluation metrics.

> Frameworks used: **PyTorch**, **scikit-learn**, **Seaborn**, **NLTK**, and **ROUGE**

---

##  1. Neural Networks Basics

**File:** `1. Neural Networks Basics.py`

This module introduces the foundation of deep learning models for NLP:
- Token embeddings using `nn.Embedding`
- Feedforward network with `ReLU` activation
- Loss calculation using `CrossEntropyLoss`
- Model training with `Adam` optimizer

**Sample Output:**
```
Epoch 1, Loss: 0.7079
Epoch 2, Loss: 0.5560
Epoch 3, Loss: 0.4319
Epoch 4, Loss: 0.3354
Epoch 5, Loss: 0.2628
```

---

##  2. Seq2Seq Models

**File:** `2. Seq2Seq Models.py`

Implements a **Sequence-to-Sequence (Seq2Seq)** model using **Encoder–Decoder** architecture:
- Encoder: processes input sequences and generates context vectors  
- Decoder: generates target sequences step by step  
- Optional: integrates teacher forcing for training stability  

**Key Concepts:**
- Encoder–Decoder structure with GRU/LSTM  
- Handling variable-length sequences  
- Teacher forcing and prediction loop  

**Use Case:** Machine Translation or Text Summarization tasks.

---

##  3. Attention Mechanism

**File:** `3. Attention Mechanism.py`

Enhances the Seq2Seq model with an **Attention Layer**, allowing the decoder to “focus” on relevant input words at each time step.

**Architecture Components:**
- `Attention`: Computes alignment scores between hidden states  
- `Encoder`: Embeds and encodes the input  
- `Decoder`: Uses attention-weighted context vectors  
- `Seq2Seq`: Combines all modules for end-to-end training  

**Output Example:**
```
Output shape: torch.Size([5, 2, 10])
```

---

##  4. CNN for Text Classification

**File:** `4. CNN for Text Classification.py`

Demonstrates how **Convolutional Neural Networks (CNNs)** can be applied to text classification.

**Highlights:**
- Embedding layer for text representation  
- 1D Convolution + MaxPooling layers for feature extraction  
- Fully connected layer for classification output  
- Often used for sentiment analysis and topic classification  

**Key Concepts:**
- Text embeddings  
- N-gram feature extraction via filters  
- Batch normalization and dropout for generalization  

---

##  5. Evaluation Metrics for NLP

**File:** `5. Evaluation Metrics.py`

Covers evaluation metrics across major NLP tasks.

###  Classification
- Accuracy, Precision, Recall, F1-score, and Confusion Matrix  
- Visualization using Seaborn  

###  Sequence Labeling (NER)
- Entity-level metrics (Precision, Recall, F1) with `seqeval`

### 📝 Text Generation
- BLEU and ROUGE scores  
- Perplexity computation  

###  Regression (Optional)
- MSE, MAE, and R² metrics  

**Example Output:**
```
Accuracy: 0.67
Precision: 0.67
Recall: 0.67
F1-score: 0.67
ROUGE Scores: {'rouge1': 0.83, 'rougeL': 0.83}
Perplexity: 8.16
```

---

##  Installation

Install all dependencies with:

```bash
pip install torch scikit-learn seaborn matplotlib nltk rouge-score seqeval
```

---

##  Learning Outcomes

After completing this series, you will be able to:
- Build and train neural networks using PyTorch  
- Understand encoder–decoder and attention mechanisms  
- Implement CNNs for text classification  
- Evaluate NLP models effectively across different tasks  

---
---
