import numpy as np
from lib.NotesCleaning import NotesCleaning

class EmbeddingUtils:
    def compute_embeddings(self, word2vec_model, sentence: str) -> list[float]:
        embed = []
        sentence = NotesCleaning().clean_sentence(sentence)
        for word in sentence:
            try:
                embed.append(word2vec_model[word])
            except:
                pass

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