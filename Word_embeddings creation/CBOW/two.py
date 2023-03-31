import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial.distance import cosine
import pandas as pd
import pickle
from collections import Counter
import os
import json
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        input_embeds = self.embeddings(inputs)
        embeds = torch.mean(input_embeds, dim=1)
        out = self.linear(embeds)
        return F.log_softmax(out, dim=1)


#load the data from sentences
with open('sentences.json') as f:
    sentences = json.load(f)


def build_vocab(sentences, min_count=5):
    freq_dist =[]
    word_counts = Counter()
    for sentence in sentences:
        for word in sentence:
            word_counts[word] += 1
            
    word_counts['<UNK>']=0
    k = 0 
    #words which occur less than min_count are replaced with <UNK> token
    for word, count in word_counts.items():
        if count < min_count:
            word_counts['<UNK>'] += count
            k+=count
        else:
            freq_dist.append(count)
    
    freq_dist.append(k)
            

    #add <UNK> to vocab 
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab.append("UNK")
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word2idx, idx2word , freq_dist

vocab, word2idx, idx2word ,freq_dist = build_vocab(sentences)

#replace words in sentences having count less than min_count with <UNK> token
for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        if word not in vocab:
            sentences[i][j] = "UNK"


window_size = 4

def build_dataset(sentences, word2idx):
    dataset = []
    for sentence in sentences:
        for i in range(window_size, len(sentence) - window_size):
            # context = sentence[i - window_size: i] + sentence[i + 1: i + window_size + 1]
            target = sentence[i]
            target_index = word2idx[target]
            context_indices = []
            for j in range(i - window_size, i + window_size + 1):
                if j != i:
                    context_indices.append(word2idx[sentence[j]])     
            dataset.append((context_indices, target_index))
    return dataset


dataset = build_dataset(sentences, word2idx)
NEG_SAMPLE_SIZE =30
def negative_sampling(dataset, word2idx, vocab_size, k=5):
    normalized_freq = F.normalize(
    torch.Tensor(freq_dist).pow(0.75), dim=0)  # p(w)^0.75
    weights = torch.ones(len(freq_dist))  # weights for each word
    for _ in tqdm(range(len(freq_dist))):
        for _ in range(NEG_SAMPLE_SIZE):
            neg_index = torch.multinomial(normalized_freq, 1)[
            0]  # sample a word
        # increase the weight of the sampled word
            weights[neg_index] += 1

    return weights

weights = negative_sampling(dataset, word2idx, len(vocab), k=5)

import torch
from torch._six import with_metaclass


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)

# mypy doesn't understand torch._six.with_metaclass
class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):  # type: ignore[misc]
    pass

from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()


BATCH_SIZE = 50
LEARNING_RATE =0.001
EMBEDDING_SIZE = 30
NUM_REVIEWS = 100
num_epochs = 20
model = CBOW(len(vocab), EMBEDDING_SIZE)
def train(num_epochs):
        losses = []
        loss_fn = nn.NLLLoss(weight=weights)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")
            total_loss = 0
            for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
                batch = dataset[i: i + BATCH_SIZE]
                

                context = [x[0] for x in batch]
                focus = [x[1] for x in batch]
                

                context_var = Variable(torch.LongTensor(context))
                focus_var = Variable(torch.LongTensor(focus))
                
                optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
                optimizer.zero_grad()
                log_probs = model(context_var)
                loss = loss_fn(log_probs, focus_var)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Loss: {loss.item()}")
            losses.append(total_loss)

train(num_epochs)

def get_embedding( word_idx):
    embedding_index = Variable(torch.LongTensor([word_idx]))
    return model.embeddings(embedding_index).data[0]

def get_closest(_word, k):
    word = _word.lower()
    if word not in vocab:
        print(
            f"{_word} not in vocabulary. Falling back to Out-of-Vocabulary Token")
        word = "UNK"
    distances = []
    focus_index = word2idx[word]
    focus_embedding = get_embedding(focus_index)
    for i in range(1, len(vocab)):
        if i == focus_index:
            continue
        comp_embedding = get_embedding(i)
        comp_word = idx2word[i]
        dist = cosine(focus_embedding, comp_embedding)
        distances.append({'Word': comp_word, 'Distance': dist})
    distances = sorted(distances, key=lambda x: x['Distance'])
    return pd.DataFrame(distances[:k])


#save the embeddings for all fords in vocab in file
import pickle
embeddings = []
for i in range(1, len(vocab)):
    embeddings.append(get_embedding(i))
with open('data.pkl', 'wb') as f:
    pickle.dump(embeddings, f)



