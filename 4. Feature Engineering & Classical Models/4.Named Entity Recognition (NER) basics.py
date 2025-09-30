import spacy
from spacy.pipeline import EntityRuler

# -------------------------------
# 1. Load pretrained spaCy NER
# -------------------------------
print("\n--- Pretrained NER Example ---")
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion in 2021. Tim Cook met the CEO in San Francisco."
doc = nlp(text)

print("Input:", text)
print("Entities detected:")
for ent in doc.ents:
    print(f"{ent.text:<15} {ent.label_}")

# -------------------------------
# 2. Add custom rule-based entities
# -------------------------------
print("\n--- Rule-based EntityRuler Example ---")

# Add rule-based entity patterns
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PRODUCT", "pattern": "iPhone 13"},
    {"label": "PRODUCT", "pattern": "iPhone"},
]
ruler.add_patterns(patterns)

custom_text = "I bought a new iPhone 13 and the phone is great."
doc2 = nlp(custom_text)

print("Input:", custom_text)
print("Entities detected:")
for ent in doc2.ents:
    print(f"{ent.text:<15} {ent.label_}")

# -------------------------------
# 3. Notes on evaluation (theory)
# -------------------------------
print("\n--- Evaluation & Next Steps ---")
print("• Evaluate NER using precision, recall, F1 (needs labeled data, e.g., CoNLL format).")
print("• For domain-specific NER, fine-tune transformer models (BERT, RoBERTa) or train spaCy pipeline.")
print("• Rule-based (regex/gazetteers) is useful when entities are predictable (e.g., product names).")
print("• Statistical models (spaCy, Hugging Face) are better for general-purpose NER.")



#output

# --- Pretrained NER Example ---
# Input: Apple is looking at buying U.K. startup for $1 billion in 2021. Tim Cook met the CEO in San Francisco.
# Entities detected:
# Apple           ORG
# U.K.            GPE
# $1 billion      MONEY
# 2021            DATE
# Tim Cook        PERSON
# San Francisco   GPE

# --- Rule-based EntityRuler Example ---
# Input: I bought a new iPhone 13 and the phone is great.
# Entities detected:
# iPhone 13       PRODUCT

# --- Evaluation & Next Steps ---
# • Evaluate NER using precision, recall, F1 (needs labeled data, e.g., CoNLL format).
# • For domain-specific NER, fine-tune transformer models (BERT, RoBERTa) or train spaCy pipeline.
# • Rule-based (regex/gazetteers) is useful when entities are predictable (e.g., product names).
# • Statistical models (spaCy, Hugging Face) are better for general-purpose NER.