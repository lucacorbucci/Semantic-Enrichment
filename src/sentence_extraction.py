import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import sys
sys.path.append('..')
from lib.doctorailib.doctorailib import doctorai
from lib.doctorXAIlib.doctorXAIlib import doctorXAI
import pickle
from lib.embedding_utils import EmbeddingUtils, Embedding, EmbeddingType
from lib.utils_similarity import UtilsSimilarity
from lib.utils import Utils
from lib.semantic_enrichment import SemanticEnrichment, WindowStore
from lib.NotesCleaning import NotesCleaning
import gensim
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as f
import numpy as np
import pandas as pd
import sent2vec
import time

logging.info("Loading preliminary files")
description_dict = pickle.load(open("../data/mapping_relations/description.pkl",'rb'))
associated_morphology_dict = pickle.load(open("../data/mapping_relations/associated_morphology.pkl",'rb'))
due_to_dict = pickle.load(open("../data/mapping_relations/due_to.pkl",'rb'))
finding_site_dict = pickle.load(open("../data/mapping_relations/finding_site.pkl",'rb'))
patient_admission_dictionary = pickle.load(open('../data/patient_admission_dictionary.pkl', 'rb'))
mapping = pickle.load(open("../data/icd9_mapping/mapping_icd9_description.pkl", "rb"))
abbreviations = pickle.load(open("../data/abbreviations/abbreviations_dict.pkl", "rb"))
logging.info("Loading the model")

# ----------------------------------------------------------- #
# Bert
# ----------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
embedding = Embedding(EmbeddingType.CLINICALBERT, model, tokenizer)

# ----------------------------------------------------------- #
# BioSentVec
# ----------------------------------------------------------- #
# model_path = '../data/embeddings/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
# model = sent2vec.Sent2vecModel()
# try:
#     model.load_model(model_path)
# except Exception as e:
#     logging.error(e)
# embedding = Embedding(EmbeddingType.BIOSENTVEC, model)


# ----------------------------------------------------------- #
# BioWordVec    
# ----------------------------------------------------------- #
# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
#     '../data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
#     binary=True,
#     limit=int(4E7)
# )
# embedding = Embedding(EmbeddingType.BIOWORDVEC, word2vec_model)


logging.info('Model successfully loaded')

# Load the dataset
# df = pd.read_csv("../data/validation/original_dataset/dataset_annotato_0.csv")
df = pd.read_csv("../data/selected_df.csv")
logging.info("CSV loaded")

se = SemanticEnrichment()

logging.info("Starting to extract sentences...")


higher, lower = se.extract_relevant_sentences(df, 
                                threshold = 95, 
                                k = 1,
                                finding_site_dict = finding_site_dict,
                                due_to_dict = due_to_dict,
                                associated_morphology_dict = associated_morphology_dict,
                                description_dict = description_dict,
                                embedding = embedding,
                                abbreviations = abbreviations)


higher = se.create_result_dataframe(higher)
lower = se.create_result_dataframe(lower)

logging.info("Writing the Csv on disk...")
higher.to_csv("../data/validation/valido_non_valido/bert.csv", index=False)
#lower.to_csv("../data/validation/valido_non_valido/biosentvec_lower_no_abbreviations.csv", index=False)    