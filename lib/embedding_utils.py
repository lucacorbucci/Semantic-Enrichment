import numpy as np
import torch
import torch.nn.functional as f
from lib.NotesCleaning import NotesCleaning
from enum import Enum

class EmbeddingType(Enum):
    BIOWORDVEC = 1
    CLINICALBERT = 2
    BIOSENTVEC = 3

class Embedding:
    def __init__(self, embedding_type, model, tokenizer = None):
        self.embedding_type = embedding_type
        self.model = model
        self.tokenizer = tokenizer
class EmbeddingUtils:

    def bert_embedding(self, sentence, bert_model, bert_tokenizer):
        texts = [" ".join(sentence)]
        encodings = bert_tokenizer(
            texts,
            padding=True,
            return_tensors='pt'
        )
        with torch.no_grad():   
            embeds = bert_model(**encodings)
        return embeds[0][0]

    def w2v_embedding(self, sentence, model):
        embed = []
        for word in sentence:
            try:
                embed.append(model[word])
            except:
                pass
        return embed

    def s2v_embedding(self, sentence, model):
        sentence_vector = model.embed_sentence(sentence)
        return sentence_vector[0]


    def compute_embeddings(self, embedding: Embedding, sentence: str):
        sentence = NotesCleaning().clean_sentence(sentence)
        embed = None
        if embedding.embedding_type == EmbeddingType.BIOWORDVEC:
            embed = self.w2v_embedding(sentence, embedding.model)
        elif embedding.embedding_type == EmbeddingType.CLINICALBERT:
            embed = self.bert_embedding(sentence, embedding.model, embedding.tokenizer)
        else:
            sentence = " ".join(sentence)
            return self.s2v_embedding(sentence, embedding.model)
        
        
        windows_embedding = np.average(
            embed, axis=0
        )

        return windows_embedding

    def compute_window_embedding(self, word2vec_model, window):
        embed = []
        for word in window:
            try:
                embed.append(word2vec_model[word])
            except:
                pass

        windows_embedding = np.average(
            embed, axis=0
        )

        return windows_embedding
