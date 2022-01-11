from pymedtermino.snomedct import *
import importlib
import lib.embedding_utils
from lib.embedding_utils import EmbeddingUtils
import re

importlib.reload(lib.embedding_utils)


class ICD9SnomedMapping:
    def extract_relations(
        self, mapping_dict, relation_name: str
    ) -> dict[str, set]:
        relation_dict = {}
        for code in mapping_dict.keys():
            relation_set = set()
            for ID in mapping_dict[code]:
                try:
                    concept = SNOMEDCT[ID]
                    if relation_name == "finding_site":
                        relations = concept.finding_site
                    elif relation_name == "causative_agent":
                        relations = concept.causative_agent
                    elif relation_name == "associated_morphology":
                        relations = concept.associated_morphology
                    elif relation_name == "due_to":
                        relations = concept.due_to
                    elif relation_name == "description":
                        relation = concept
                        relation_str = str(relation)
                        start = relation_str.find("#")
                        relation_str = relation_str[start + 1 :]
                        relation_str = re.sub(
                            "[\(\[].*?[\)\]]", "", relation_str
                        ).strip()
                        relation_str = relation_str.lower()
                        relation_set.add(relation_str)
                        relations = []
                    else:
                        break

                    if relations and relation_name != "description":
                        for relation in relations:
                            relation_str = relation.__str__()
                            start = relation_str.find("#")
                            end = relation_str.find("(")
                            relation_str = relation_str[start + 1 : end].strip()
                            relation_str = relation_str.lower()
                            relation_set.add(relation_str)
                    if relation_set:
                        relation_dict[code] = relation_set
                except:
                    pass
        return relation_dict

    def compute_relation_embeddings(
        self, mapping_dict, model
    ):
        embedding_dict = {}
        for code in mapping_dict.keys():
            relations = list(mapping_dict[code])
            for relation in relations:
                embedding_dict[relation] = EmbeddingUtils().compute_embeddings(
                    model, relation
                )
        return embedding_dict