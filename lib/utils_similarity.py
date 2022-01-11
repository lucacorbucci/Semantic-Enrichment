import numpy as np
from lib.embedding_utils import EmbeddingUtils, EmbeddingType
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


class UtilsSimilarity:

    """"""

    def compute_cosine_similarity(self, windows_embedding, embedding_description):
        try:
            similarity = cosine_similarity(
                [windows_embedding], [embedding_description]
            )[0][0]
        except:
            return -1

        return similarity

    def rolling_window_embedding(self, note, window_size: int, embedding):
        """Summary or Description of the Function

        Parameters:
        df (DataFrame): The dataset where we stored all data about patients and visits

        Returns:
        list[list[str]]: The list of all the ICD 9 assigned to the patients in multiple admissions to the hospital

        """
        x = pd.Series(note)
        windows_size = window_size if window_size < len(x) else len(x)
        windows = sliding_window_view(x, windows_size)
        list_windows = []
        embedded_windows = []

        for window in windows:
            if embedding.embedding_type == EmbeddingType.BIOWORDVEC:
                embedded_windows.append(
                    EmbeddingUtils().compute_window_embedding(embedding.model, window)
                )
            else:
                joined_window = " ".join(window)
                embedded_windows.append(
                    EmbeddingUtils().compute_embeddings(embedding, joined_window)
                )
            
            
            list_windows.append(window)
        return list_windows, embedded_windows

    

    def find_most_similar_part_in_the_note(self, df, model, relation_dict):
        codes = []
        windowed_notes = []
        for index, row in df.iterrows():
            note = eval(row.Token)
            list_windows, embedded_windows = self.rolling_window_embedding(
                note, 30, model
            )
            for ICD9 in eval(row.ICD9_CODE):
                best_similarity = 0
                best_window = []
                best_embedding = []
                for window, embedding_window in zip(list_windows, embedded_windows):
                    try:
                        relation = list(relation_dict[ICD9])[0]
                        relation_embedding = EmbeddingUtils().compute_window_embedding(
                            model, relation
                        )
                    except:
                        relation_embedding = np.array([])

                    if relation_embedding.size != 0:
                        similarity = self.compute_cosine_similarity(
                            embedding_window, relation_embedding
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_window = window
                if len(best_window) != 0:
                    codes.append(ICD9)
                    windowed_notes.append(best_window.copy())
        return codes, windowed_notes

    def find_best_window(
        self,
        codes,
        windowed_notes,
        relation_dict,
        relation_embedding_dict,
        model,
        min_window,
        max_window,
    ):
        all_similarities = {}
        for code, note in zip(codes, windowed_notes):
            try:
                relation = list(relation_dict[code])[0]
                relation_embedding = relation_embedding_dict[relation]
            except:
                relation_embedding = np.array([])
            if relation_embedding.size != 0:
                for window_size in range(min_window, max_window):
                    list_windows, embedded_windows = self.rolling_window_embedding(
                        note, window_size, model
                    )
                    best_similarity = 0
                    for window, embedding_window in zip(list_windows, embedded_windows):
                        if relation_embedding.size != 0:
                            similarity = self.compute_cosine_similarity(
                                embedding_window, relation_embedding
                            )
                            if similarity > best_similarity:
                                best_similarity = similarity
                    if all_similarities.get(window_size, None):
                        all_similarities[window_size].append(best_similarity)
                    else:
                        all_similarities[window_size] = [best_similarity]
        return all_similarities


    def compute_mean_per_window_size(self, all_similarities: dict):
        size = []
        mean = []
        for window_size in all_similarities.keys():
            similarity_values = all_similarities.get(window_size)
            size.append(window_size)
            similarity_values = [value for value in similarity_values if value != 0]
            mean.append(np.mean(similarity_values))
        return size, mean

    
    def plot_window_analysis(self, window_length, mean_for_plot, relation):
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 7)
        ax.plot(window_length, mean_for_plot)
        plt.xlabel("Window size")
        plt.ylabel("Average Similarity")
        plt.title(
            f"Similarity between the {relation} relation and the note when varying the window size"
        )
        plt.xticks(np.arange(3, 30, 1.0))
        plt.tight_layout()
        plt.show()