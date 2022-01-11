<div align="center">
  <h1> 🗒 Semantic Enrichment of XAI Explanation for Healthcare 👨‍⚕️ </h1>

  <p>
    <strong> Semantically enrich the explanations of Doctor XAI </strong>
  </p>

</div>

## Table of Content

- Introduction
- Install procedure
- How to Use

## Introduction

Being able to explain black-box models decisions is crucial to increase doctors trust in AI-based clinical decision support systems.However, already existing eXplainable Artificial Intelligence (XAI) techniques can provide explanations that are not easily understand-able to experts outside of AI. We present a methodology that aims to enable clinical reasoning by semantically enrich the explanationof a prediction by exploiting the content of a medical ontology and clinical notes. We validate our methodology with the support of adomain expert, showing that this method could be a first step toward developing a natural language explanation of black-box models.

## Install Procedure

To use our method, you need to download the following things:

- [Doctor XAI](https://github.com/CeciPani/DrXAI)
- [Mimic III dataset](https://physionet.org/content/mimiciii/1.4/)
- [Snomed-CT medical ontoloty](https://www.nlm.nih.gov/healthit/snomedct/international.html)
- [BioWordVec](https://github.com/ncbi-nlp/BioWordVec)

## How to Use

[This notebook](https://github.com/lucacorbucci/Semantic_enrichment_of_xai_explanations_for_healthcare/blob/master/example/extract_sentece.ipynb) shows an example of our methodology:

- We load the data
- We extract the relevant ICD-9 using Doctor XAI
- We use our methodology to extract the relevant sentences.

The output of our method is a pandas dataframe which contains the following columns:

- relation_type: the type of the extracted relation
- relation: the snomed relation we had to extract
- extracted_substring: the sentence of the clinical note that has the highest similarit with the relation
- converted_string: the original sentence taken from the clinical note without note preprocessing
- similarity: the similarity value between the two sentences
