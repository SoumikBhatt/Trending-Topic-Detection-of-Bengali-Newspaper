import nltk
from nltk import FreqDist
import string
import matplotlib.pyplot as plt
import pandas as pd
import re

FileName = (r"Test_Title1.txt")
with open(FileName, 'r',encoding='utf-8-sig') as file:
     lines_in_file = file.read()

nltk_tokens = nltk.word_tokenize(lines_in_file)

# Remove Punctuation
nltk_tokens=[x for x in nltk_tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]

# Stopwords
stopwords = set(line.strip() for line in open('stopwords.txt',encoding='utf-8-sig'))
stopwords = stopwords.union(set(['‘','’']))

filtered_sentence = [w for w in nltk_tokens if not w in stopwords]

filtered_sentence = []



for w in nltk_tokens:
     if w not in stopwords:
           filtered_sentence.append(w)

n_print = int(input("How many most common words to print: "))
print('Tokenize Text: ')
print(nltk_tokens)
print("\nFiltered Sentence: ")
print(filtered_sentence)
bb=list(nltk.bigrams(filtered_sentence))
print("\nBigram Texts: ")
print(bb)
fdist1=FreqDist(bb)
frequent_bigram = fdist1.most_common(n_print)
print("\nMost Frequent Bigrams: ")
print(frequent_bigram)

lst = frequent_bigram
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')
#plt.rc(prop)
plt.rc('font', **{'sans-serif' : 'Kalpurush', 'family' : 'sans-serif'})
plt.show()
