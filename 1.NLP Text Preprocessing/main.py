import nltk

# =============================
# Download Required NLTK Data
# =============================
nltk.download('punkt')
nltk.download('punkt_tab')   # 🔥 Required for sentence tokenization
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# =============================
# Input Text
# =============================
text = "Natural Language Processing (NLP) is a sub-field of Artificial Intelligence."

# =============================
# Sentence Tokenization
# =============================
sentences = sent_tokenize(text)

# =============================
# Word Tokenization
# =============================
words = word_tokenize(text)

# =============================
# Stopword Removal
# =============================
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]

# =============================
# Stemming
# =============================
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in filtered_words]

# =============================
# Lemmatization
# =============================
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]

# =============================
# Save Results to a TXT File
# =============================
output_file = "nlp_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Input Text:\n")
    f.write(text + "\n\n")
    f.write("Sentences:\n")
    f.write(str(sentences) + "\n\n")
    f.write("Words:\n")
    f.write(str(words) + "\n\n")
    f.write("Filtered Words:\n")
    f.write(str(filtered_words) + "\n\n")
    f.write("Stems:\n")
    f.write(str(stemmed) + "\n\n")
    f.write("Lemmas:\n")
    f.write(str(lemmatized) + "\n")

print(f"✅ Results saved in {output_file}")
