import pandas as pd
from lib.doctorXAIlib.doctorXAIlib import doctorXAI
from lib.embedding_utils import EmbeddingUtils
from lib.utils_similarity import UtilsSimilarity
from lib.NotesCleaning import NotesCleaning
import numpy as np
import re


class WindowStore:
    def __init__(self, similarity, best_substring):
        self.similarity = similarity
        self.best_substring = best_substring
        self.relation = None

    def update(self, similarity, best_substring):
        self.similarity = similarity
        self.best_substring = best_substring

    def add_relation(self, relation):
        self.relation = relation

    def __repr__(self):
        return f"{self.relation}, {self.similarity}, {self.best_substring}"

    def __str__(self):
        return f"{self.relation}, {self.similarity}, {self.best_substring}"


class SemanticEnrichment:
    def __init__(
        self,
        dataset_sequences,
        black_box_oracle,
        ontology_path_file,
        ICD9_description_dict,
        CCS_description_dict,
        admission_mimic_sequences,
    ):
        self.dataset_sequences = dataset_sequences
        self.black_box_oracle = black_box_oracle
        self.ontology_path_file = ontology_path_file
        self.ICD9_description_dict = ICD9_description_dict
        self.CCS_description_dict = CCS_description_dict
        self.admission_mimic_sequences = admission_mimic_sequences

    def get_all_ICD9_codes(self, df: pd.DataFrame, patient_id: int) -> list[list[str]]:
        codes = []
        for index, row in df[df.SUBJECT_ID == patient_id].iterrows():
            codes.append(eval(row.ICD9_CODE))

        return codes

    def explain_and_get_most_relevant_ICD9(
        self, df_dictionary, patient_id: int
    ) -> list[str]:
        """Summary or Description of the Function

        Parameters:
        df (DataFrame): The dataset where we stored all data about patients and visits

        Returns:
        list[list[str]]: The list of all the ICD 9 assigned to the patients in multiple admissions to the hospital

        """
        patient_seq = df_dictionary[patient_id][1]
        dr_xai = doctorXAI.DoctorXAI(
            patient_sequence=patient_seq,
            dataset_sequences=self.dataset_sequences,
            black_box_oracle=self.black_box_oracle,
            ontology_path_file=self.ontology_path_file,
            syn_neigh_size=500,
        )
        _, _, list_split_conditions_ICD9, _, _, _, _, _, _ = dr_xai.extract_rule(
            ICD9_description_dict=self.ICD9_description_dict,
            CCS_description_dict=self.CCS_description_dict,
        )

        return [
            condition.split(" <=")[0].split(" >")[0].replace(".", "")
            for condition in list_split_conditions_ICD9
        ]

    def get_ICD9_with_dot(self, df):
        """Summary or Description of the Function

        Parameters:
        df (DataFrame): The dataset where we stored all data about patients and visits

        Returns:
        list[list[str]]: The list of all the ICD 9 assigned to the patients in multiple admissions to the hospital

        """
        viewed = set()
        dictionary = {}
        for _, row in df.iterrows():
            subject_id = row.SUBJECT_ID
            if subject_id not in viewed:
                viewed.add(subject_id)
                hadm_id = int(row.HADM_ID)
                for index, list in enumerate(self.admission_mimic_sequences):
                    for item in list:
                        if item == hadm_id:
                            dictionary[subject_id] = (
                                list,
                                self.dataset_sequences[index],
                            )
        return dictionary

    def get_relevant_HADM_ID(self, df, subject_id, relevant_ICD9):
        """Summary or Description of the Function

        Parameters:
        df (DataFrame): The dataset where we stored all data about patients and visits

        Returns:
        list[list[str]]: The list of all the ICD 9 assigned to the patients in multiple admissions to the hospital

        """
        relevant_HADM_ID = []
        subject_ids = []
        for _, row in df[df.SUBJECT_ID == subject_id].iterrows():
            for code in eval(row.ICD9_CODE):
                if code in relevant_ICD9 and int(row.HADM_ID) not in relevant_HADM_ID:
                    relevant_HADM_ID.append(int(row.HADM_ID))
                    subject_ids.append(int(row.SUBJECT_ID))
        return relevant_HADM_ID, subject_ids

    def get_text_and_tokens(self, df, relevant_HADM_ID):
        """Summary or Description of the Function

        Parameters:
        df (DataFrame): The dataset where we stored all data about patients and visits

        Returns:
        list[list[str]]: The list of all the ICD 9 assigned to the patients in multiple admissions to the hospital

        """
        text = []
        token = []
        icd9 = []
        for hadm_code in relevant_HADM_ID:
            text.append(df[df.HADM_ID == hadm_code].TEXT.values[0])
            token.append(eval(df[df.HADM_ID == hadm_code].Token.values[0]))
            icd9.append(eval(list(df[df.HADM_ID == hadm_code].ICD9_CODE.values)[0]))
        return text, token, icd9

    def get_most_similar_substring(self, embedding_relation, token, window_size, model):
        similarity_substring = -1
        best_substring = ""

        list_windows, embedding_windows = UtilsSimilarity().rolling_window_embedding(
            token, window_size, model
        )

        for window, embedding_window in zip(list_windows, embedding_windows):
            similarity = UtilsSimilarity().compute_cosine_similarity(
                embedding_window, embedding_relation
            )
            if similarity > similarity_substring:
                similarity_substring = similarity
                best_substring = window
        return WindowStore(similarity_substring, best_substring)

    def get_k_most_similar_substring(
        self,
        embedding_relation,
        token,
        window_size,
        model,
        k,
    ):
        all_similarity_substring = []
        list_windows, embedding_windows = UtilsSimilarity().rolling_window_embedding(
            token, window_size, model
        )

        for window, embedding_window in zip(list_windows, embedding_windows):
            similarity = UtilsSimilarity().compute_cosine_similarity(
                embedding_window, embedding_relation
            )
            all_similarity_substring.append((similarity, window))

        all_similarity_substring.sort(key=lambda x: x[0], reverse=True)
        ans = []
        for item in all_similarity_substring[0:k]:
            ans.append(WindowStore(item[0], item[1]))
        return ans

    def extract_most_similar_part(
        self,
        relation_dict,
        notes,
        tokens,
        ICD_9,
        relevant_ICD9,
        word2vec_model,
        window_size,
        threshold,
        k,
    ):
        best_similarity = []
        for note, token, icd9_list in zip(notes, tokens, ICD_9):
            if window_size < len(token):
                best_similarity_notes = []
                for code in icd9_list:
                    best_similarity_codes = []
                    if code in relevant_ICD9:
                        relations = relation_dict.get(str(code), None)
                        if relations:
                            # We can have multiple relation for each ICD-9 code
                            for relation in relations:
                                embedding_relation = (
                                    EmbeddingUtils().compute_embeddings(
                                        word2vec_model, relation
                                    )
                                )
                                best_window_substring = (
                                    self.get_k_most_similar_substring(
                                        embedding_relation,
                                        token,
                                        window_size,
                                        word2vec_model,
                                        k,
                                    )
                                )
                                for item in best_window_substring:
                                    item.add_relation(relation)
                                    # Here we store all the relation with the corresponding similarity value and the best substring we extracted
                                    best_similarity_codes.append(item)

                            # We compute the percentile to remove from the list the strings with a simialarity lower than this value
                            similarities = []
                            for item in best_similarity_codes:
                                similarities.append(item.similarity)
                            similarities = sorted(similarities)
                            percentile = np.percentile(similarities, threshold)
                            best_similarity_codes = [
                                item
                                for item in best_similarity_codes
                                if item.similarity >= percentile
                            ]

                            # For each code we store the extracted substring
                            best_similarity_notes.append((code, best_similarity_codes))
                # For each note we store all the extracted substrings with the similarity value
                best_similarity.append((note, best_similarity_notes))

        return best_similarity

    def extract_similarity_lower_than_threshold(
        self,
        relation_dict,
        notes,
        tokens,
        ICD_9,
        relevant_ICD9,
        word2vec_model,
        window_size,
        threshold,
        k,
    ):
        best_similarity = []
        for note, token, icd9_list in zip(notes, tokens, ICD_9):
            if window_size < len(token):
                best_similarity_notes = []
                for code in icd9_list:
                    best_similarity_codes = []
                    if code in relevant_ICD9:
                        relations = relation_dict.get(str(code), None)
                        if relations:
                            # We can have multiple relation for each ICD-9 code
                            for relation in relations:
                                embedding_relation = (
                                    EmbeddingUtils().compute_embeddings(
                                        word2vec_model, relation
                                    )
                                )
                                best_window_substring = (
                                    self.get_k_most_similar_substring(
                                        embedding_relation,
                                        token,
                                        window_size,
                                        word2vec_model,
                                        k,
                                    )
                                )
                                for item in best_window_substring:
                                    item.add_relation(relation)
                                    # Here we store all the relation with the corresponding similarity value and the best substring we extracted
                                    best_similarity_codes.append(item)

                            # We compute the percentile to remove from the list the strings with a simialarity lower than this value
                            similarities = []
                            for item in best_similarity_codes:
                                similarities.append(item.similarity)
                            similarities = sorted(similarities)
                            percentile = np.percentile(similarities, threshold)
                            best_similarity_codes = [
                                item
                                for item in best_similarity_codes
                                if item.similarity < percentile
                            ]

                            # For each code we store the extracted substring
                            best_similarity_notes.append((code, best_similarity_codes))
                # For each note we store all the extracted substrings with the similarity value
                best_similarity.append((note, best_similarity_notes))

        return best_similarity

    def convert_to_original_substring(self, note, substring):
        start_index_list = [a.start() for a in list(re.finditer(substring[0], note))]
        end_index_list = [a.end() for a in list(re.finditer(substring[-1], note))]
        found = False
        range_list = []
        converted_string = ""
        for start_index in start_index_list:
            for end_index in end_index_list:
                if start_index <= end_index:
                    tmp_substring = note[start_index:end_index]
                    range_list.append(
                        (tmp_substring, len(tmp_substring), start_index, end_index)
                    )
        range_list.sort(key=lambda x: x[1])
        start_converted = 0
        end_converted = 0
        converted_string = ""
        for tmp_substring, _, start, end in range_list:
            cleaned_note = NotesCleaning().clean_note(tmp_substring)

            if not found and all(word in cleaned_note for word in substring):
                converted_string = tmp_substring
                start_converted = start
                end_converted = end
                found = True

        return converted_string, start_converted, end_converted