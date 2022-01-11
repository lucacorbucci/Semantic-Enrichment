import pandas as pd
from lib.doctorXAIlib.doctorXAIlib import doctorXAI
from lib.doctorailib.doctorailib import doctorai
from lib.embedding_utils import EmbeddingUtils
from lib.utils_similarity import UtilsSimilarity
from lib.NotesCleaning import NotesCleaning
import numpy as np
import re
import pickle


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
    ):
        self.model_file = (
            "../doctorXAI/models/trained_doctorAI_output/2020_9_30_MIMIC_III_.44.npz"
        )
        self.black_box_oracle = doctorai.DoctorAI(
            modelFile=self.model_file,
            ICD9_to_int_dict="../doctorXAI/preprocessing_doctorai/ICD9_to_int_dict",
            CCS_to_int_dict="../doctorXAI/preprocessing_doctorai/CCS_to_int_dict",
        )
        self.dataset_sequences = np.load(
            "../doctorXAI/preprocessing_doctorai/mimic_sequences.npy", allow_pickle=True
        )
        self.ontology_path_file = "../lib/doctorXAIlib/ICD9_ontology.csv"
        self.ICD9_description_dict = pickle.load(
            open("../doctorXAI/ICD9_description_dict.pkl", "rb")
        )
        self.CCS_description_dict = pickle.load(
            open("../doctorXAI/CCS_description_dict.pkl", "rb")
        )
        self.admission_mimic_sequences = np.load(
            "../doctorXAI/preprocessing_doctorai/admission_mimic_sequences.npy",
            allow_pickle=True,
        )
        self.date_mimic_sequences = np.load(
            "../doctorXAI/preprocessing_doctorai/date_mimic_sequences.npy",
            allow_pickle=True,
        )

    def get_notes_to_validate(self, 
            subject_ids, 
            df, 
            admission_dictionary, 
            mapping):
        '''
        
        '''
        notes = []
        hadm_ids = []
        codes = []
        description_codes = []
        sub_ids = []
        

        for id in subject_ids:
            # We get the relevant ICD9 codes for the patient and we use these codes and the relations taken from Snomed to 
            # extract the most similar parts of the notes
            relevant_ICD9 = self.explain_and_get_most_relevant_ICD9(admission_dictionary, id)
            relevant_HADM_ID, subject_ids = self.get_relevant_HADM_ID(df, id, relevant_ICD9)
            text, _, ICD_9 = self.get_text_and_tokens(df, relevant_HADM_ID)
            
            for hadm_id, sub_id, note, icd9_list in zip(relevant_HADM_ID, subject_ids, text, ICD_9):
                for code in icd9_list:
                    if code in relevant_ICD9:
                        notes.append(note)
                        sub_ids.append(sub_id)
                        hadm_ids.append(hadm_id)
                        codes.append(code)
                        try:
                            description_codes.append(mapping[str(code)])
                        except:
                            description_codes.append("")
            break

        list_tuples = list(zip(sub_ids, hadm_ids, codes, description_codes, notes))
        notes_to_validate = pd.DataFrame(list_tuples, columns=['subject_id', 'hadm_id', 'codes', 'description', 'notes']) 

        return notes_to_validate

    def explain_and_get_most_relevant_ICD9(
        self, df_dictionary, patient_id: int
    ):
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

    def extract_relevant_sentences(self, 
                                    df, 
                                    threshold, 
                                    k, 
                                    finding_site_dict,
                                    due_to_dict,
                                    associated_morphology_dict,
                                    description_dict,
                                    embedding,
                                    abbreviations = None):
        ICD_9_extracted = df['codes']
        hadm_id_extracted = df['hadm_id']
        id_extracted = df['subject_id']
        notes_extracted, original_note_list = self.get_notes(df, hadm_id_extracted, abbreviations)
        results = []
        results_low = []
        for code, id, hadm_id, note, token in zip(ICD_9_extracted, id_extracted, hadm_id_extracted, original_note_list, notes_extracted):
            
            best_substrings_finding_site, low_substrings_finding_site = self.extract_most_similar_part(finding_site_dict, note, token, code, embedding, 7, threshold, k)
            #print(best_substrings_finding_site, low_substrings_finding_site)
            best_substrings_due_to, low_substrings_due_to = self.extract_most_similar_part(due_to_dict, note, token, code, embedding, 9, threshold, k)
            #print(best_substrings_due_to, low_substrings_due_to)
            best_substrings_associated_morphology, low_substrings_associated_morphology = self.extract_most_similar_part(associated_morphology_dict, note, token, code, embedding, 7, threshold, k)
            #print(best_substrings_associated_morphology, low_substrings_associated_morphology)
            best_substrings_description, low_substrings_description = self.extract_most_similar_part(description_dict, note, token, code, embedding, 10, threshold, k)
            #print(best_substrings_description, low_substrings_description)

            # Higher than threshold
            for best_substring in best_substrings_finding_site:
                converted_string, _, _ = self.convert_to_original_substring(note, best_substring.best_substring)
                results.append((id, hadm_id, code, "Finding_site", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            for best_substring in best_substrings_due_to:
                converted_string, _, _= self.convert_to_original_substring(note, best_substring.best_substring)
                results.append((id, hadm_id, code, "Due_to", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            for best_substring in best_substrings_associated_morphology:
                converted_string, _, _= self.convert_to_original_substring(note, best_substring.best_substring)
                results.append((id, hadm_id, code, "Associated_morphology", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            for best_substring in best_substrings_description:
                converted_string, _, _ = self.convert_to_original_substring(note, best_substring.best_substring)
                results.append((id, hadm_id, code, "Description", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))

            # Lower than threshold
            for best_substring in low_substrings_finding_site:
                converted_string, _, _ = self.convert_to_original_substring(note, best_substring.best_substring)
                results_low.append((id, hadm_id, code, "Finding_site", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            for best_substring in low_substrings_due_to:
                converted_string, _, _= self.convert_to_original_substring(note, best_substring.best_substring)
                results_low.append((id, hadm_id, code, "Due_to", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            for best_substring in low_substrings_associated_morphology:
                converted_string, _, _= self.convert_to_original_substring(note, best_substring.best_substring)
                results_low.append((id, hadm_id, code, "Associated_morphology", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            for best_substring in low_substrings_description:
                converted_string, _, _ = self.convert_to_original_substring(note, best_substring.best_substring)
                results_low.append((id, hadm_id, code, "Description", best_substring.best_substring, best_substring.similarity, best_substring.relation, converted_string))
            

        return results, results_low

    def convert_to_original_substring(self, note, substring):
        end_index_list = []
        start_index_list = []
        try:
            start_index_list = [a.start() for a in list(re.finditer(substring[0], note))]
            end_index_list = [a.end() for a in list(re.finditer(substring[-1], note))]
        except:
            return "", 0, 0

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

    def create_result_dataframe(self, results):
        # Create the dataframe with all the extracted relations
        subject_ids = []
        hadm_ids = []
        icd9_list = []
        relations_type = []
        substrings = []
        similarity = []
        relations = []
        converted_string = []
        
        for item in results:
            subject_ids.append(item[0])
            hadm_ids.append(item[1])
            icd9_list.append(item[2])
            relations_type.append(item[3])
            substrings.append(item[4])
            similarity.append(item[5])
            relations.append(item[6])
            converted_string.append(item[7])
            

        list_tuples = list(zip(subject_ids, hadm_ids, icd9_list, relations_type, relations, similarity, substrings, converted_string))

        results = pd.DataFrame(list_tuples, columns=['subject_ID', 'hadm_id', 'icd_9', 'relation_type', 'relation', 'similarity', 'extracted_substring', 'converted_string'])  
        return results   

    # *************************************************************************************************************************


    def get_all_ICD9_codes(self, df: pd.DataFrame, patient_id: int):
        codes = []
        for index, row in df[df.SUBJECT_ID == patient_id].iterrows():
            codes.append(eval(row.ICD9_CODE))

        return codes

    

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



    def append_results(self, results, relevant_HADM_ID, best_relation, type, id):
        for HADM_ID, best_substring in zip(relevant_HADM_ID, best_relation):
            results.append((id, HADM_ID, type, best_substring[1]))

        return results

   
    def get_k_most_similar_substring(
        self,
        embedded_relation,
        window_size,
        embedding,
        k,
        token
    ):
        all_similarity_substring = []
        list_windows, embedding_windows = UtilsSimilarity().rolling_window_embedding(
                token, window_size, embedding
            )

        for window, embedding_window in zip(list_windows, embedding_windows):
            similarity = UtilsSimilarity().compute_cosine_similarity(
                embedding_window, embedded_relation
            )
            all_similarity_substring.append((similarity, window))

        all_similarity_substring.sort(key=lambda x: x[0], reverse=True)
        ans = []
        for item in all_similarity_substring[0:k]:
            ans.append(WindowStore(item[0], item[1]))
        return ans
        

    def get_notes(self, df, hadm_id_extracted, abbreviations = None):
        notes_extracted = []
        for had in hadm_id_extracted:
            notes_extracted.append(df[df['hadm_id'] == had].notes.values[0])
        notes_extracted_cleaned = []
        original_note_list = []
        for note in notes_extracted:
            if abbreviations:
                _, _, cleaned_note_splitted = NotesCleaning().clean_note_and_remove_abbreviations(note, abbreviations)
                cleaned_note = " ".join(cleaned_note_splitted)
            else:
                cleaned_note = NotesCleaning().clean_note(note)
            notes_extracted_cleaned.append(NotesCleaning().split_notes(cleaned_note))
            original_note_list.append(note)

        notes_extracted = notes_extracted_cleaned
        return notes_extracted, original_note_list

