#import libraries 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
# import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
#import libraries 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from numpy.linalg import norm
import scipy as sp
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk import bigrams
import itertools
import torch
import torch


#load the data from reviews_Movies_and_TV.json file using for loop to iterate through 40,000 sentences in a loop and append it to a list
data = []
with open('reviews_Movies_and_TV.json') as f:
    count = 0 
    for line in f:
        count = count + 1
        data.append(json.loads(line))
        if count == 80000:
            break
# print(len(data))

# extract the review text from data and append it to a list
review_text = []
for i in range(len(data)):
    review_text.append(data[i]['reviewText'])
print((review_text[1000:1531]))

#tokenize the review text using into sentences
sentences = []
for i in range(len(review_text)):
    sentences.append(word_tokenize(review_text[i]))
# print(sentences[1000:1531])

#remove the stop words and punctuations from the sentences
stop_words = set(stopwords.words('english'))
for i in range(len(sentences)):
    sentences[i] = [re.sub(r'[^\w\s]', '', word) for word in sentences[i] if word not in stop_words and word != '']
    # sentences[i] = [word for word in sentences[i] if word not in stop_words]
# print(sentences[1000:1531])

#save the sentences to a file in json format
with open('sentences.json', 'w') as f:
    json.dump(sentences, f)




#load the sentences from the file sentences.json
with open('sentences.json') as f:
    saved_sentences = json.load(f)

tokens = saved_sentences


#claculate the frequency of each word
word_freq = {}
for i in range(len(tokens)):
        for word in tokens[i]:
                if word not in word_freq:
                        word_freq[word] = 1
                else:
                        word_freq[word] += 1   

#remove words that has word frequency less than 5 from tokens
frequent_tokens = []
for i in range(len(tokens)):
        frequent_tokens.append([word for word in tokens[i] if word_freq[word] > 4])
        

vocab = list(set([word for sentence in tokens for word in sentence]))
vocab_size = len(vocab)
word2idx = {word:idx for idx, word in enumerate(vocab)}
idx2word = {idx:word for idx, word in enumerate(vocab)}
# co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
#define co_matrix as scapy sparse matrix

# producea co-occurence matrix
def co_occurence_matrix(tokens, window_size):
        co_matrix = sp.sparse.lil_matrix((vocab_size, vocab_size))
        for sentence in tokens:
                indices = [word2idx[word] for word in sentence]
                for center_word_pos in range(len(indices)):
                        for w in range(-window_size, window_size + 1):
                                context_word_pos = center_word_pos + w
                                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                                        continue
                                context_word_idx = indices[context_word_pos]
                                co_matrix[indices[center_word_pos], context_word_idx] += 1
                                # co_matrix[context_word_idx,indices[center_word_pos]] += 1

        return co_matrix

co_matrix = co_occurence_matrix(tokens, 4)


#convert co_matrix to a sparse matrix
co_matrix = sp.sparse.lil_matrix(co_matrix)
co_matrix = co_matrix.astype(float)

#perform singular value decomposition using spacy
u, s, v = sp.sparse.linalg.svds(co_matrix , k=100)


dimension = 0
cutoff_sv = 0.5
denominator = sum(s)
cont_sum = 0
for i, x in enumerate(s):
    cont_sum += x
    if (cont_sum/denominator > cutoff_sv):
        dimension = i+1
        break
vectors = u[:, :dimension]
vectors = [v/norm(v) for v in vectors]

f = open("svd_vectors.txt", "w")
for word, vector in zip(vocab, vectors):
        f.write(word)
        f.write('\n')
        p = str(vector).replace('\n', '')
        f.write(p)
        f.write('\n')
f.close()


def get_top_tenn(word):
    vec_words = []
    line1, line2 = (1, 1)
    vector=[]
    with open("svd_vectors.txt", "r") as f:
        for line1 in f:
            line1 = line1[:-1]
            line2 = f.readline()
            line2 = re.sub(r' +', ',', line2).replace('[,', '[')
            try:
                line2 = eval(line2)
                # print("word", line1, "vector", line2)
                vec_words += [(line1, line2)]
                if (line1 == word):
                    vector = line2
            except:
            #   print("error")
              f.readline()

    dist={}
    # print(vector)
    for vec_word in vec_words:
        if(vec_word[0] == word):
            continue
        try:
            dist[vec_word[0]] = np.dot(vector, vec_word[1])
        except:
            # print("error")
            pass
    sort_v = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    #print only top ten words 
    return sort_v[:10]
    # for i in range(10):
    #     print(sort_v[i])



words =["child","charming", "win","excellent","car"]

vector_child = []
vector_charming = []
vector_win = []
vector_excellent = []
vector_car = []


ten_child = []
ten_charming = []
ten_win = []
ten_excellent = []
ten_car = []

ten_child = get_top_tenn("child")
ten_charming = get_top_tenn("charming")
ten_win = get_top_tenn("win")
ten_excellent = get_top_tenn("excellent")
ten_car = get_top_tenn("car")


f = open("top_ten_words.txt", "w")
f.write("child")
f.write('\n')
for i in range(10):
    f.write(str(ten_child[i]))
    f.write('\n')
f.write("charming")
f.write('\n')
for i in range(10):
    f.write(str(ten_charming[i]))
    f.write('\n')
f.write("win")
f.write('\n')
for i in range(10):
    f.write(str(ten_win[i]))
    f.write('\n')
f.write("excellent")
f.write('\n')
for i in range(10):
    f.write(str(ten_excellent[i]))
    f.write('\n')
f.write("car")
f.write('\n')
for i in range(10):
    f.write(str(ten_car[i]))
    f.write('\n')
f.close()


ten_Titanic = []
vector_Titanic = []

ten_Titanic = get_top_tenn("Titanic")
pca = PCA(n_components=2)
vector_Titanic = pca.fit_transform(vector_Titanic)
