from collections import Counter, defaultdict
import re

text = "I love AI. AI is the future. I love learning AI."
tokens = re.findall(r"\w+", text.lower())
unigrams = Counter(tokens)
bigrams = Counter(zip(tokens, tokens[1:]))

# bigram conditional probability P(next | current)
cond_prob = {}
for (w1,w2), cnt in bigrams.items():
    cond_prob.setdefault(w1, {})[w2] = cnt / unigrams[w1]

print("Unigrams:", unigrams)
print("Bigrams:", bigrams)
print("P('ai'|'love') =", cond_prob.get('love', {}).get('ai'))


#result

# Unigrams: Counter({'ai': 3, 'i': 2, 'love': 2, 'is': 1, 'the': 1, 'future': 1, 'learning': 1})
# Bigrams: Counter({('i', 'love'): 2, ('love', 'ai'): 1, ('ai', 'ai'): 1, ('ai', 'is'): 1, ('is', 'the'): 1, ('the', 'future'): 1, ('future', 
# 'i'): 1, ('love', 'learning'): 1, ('learning', 'ai'): 1})
# P('ai'|'love') = 0.5