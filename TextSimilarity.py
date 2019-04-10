#!/usr/bin/env python
# coding: utf-8

import spacy
from nltk.corpus import wordnet as wn
import numpy as np
import re


class TextSimilarity:
    
    def __init__(self, tags_dict = None, correlation_matrix = None):
        
        self._tags = self._get_tags_dict() if tags_dict is None else tags_dict
        self._no_of_tags = len(self._tags) 
        self._tag_correlation_matrix = np.identity(self._no_of_tags) if correlation_matrix is None else correlation_matrix
        self._parser = spacy.load("en")
        
    def _get_tags_dict(self):
        with open("tags.txt","r") as fl:
            data = fl.read()

        tags = [each.strip("\n").strip() for each in re.findall(r"\n[a-z]*\s",data)]
        tags.extend(["acl","ROOT"])
        tags = sorted(tags)
        tags = {each[1]:each[0] for each in enumerate(tags,start=0)}
        return tags
    
    def _similarity_word(self, pair_A, pair_B):

        #getting head and dependent texts 
        head_a, head_b = pair_A[0].text, pair_B[0].text
        dep_a, dep_b = pair_A[2].text, pair_B[2].text

        if head_a == head_b:
            head = 1
        else:
            try:
                #WordNet synsets for heads
                head_a, head_b = wn.synsets(head_a)[0], wn.synsets(head_b)[0]

                #path based similarity (Li et. al) for head
                head = head_a.path_similarity(head_b)

                head = 0 if head is None else head  

            except Exception:
                head = 0

        if dep_a == dep_b:
            dep = 1
        else:
            try:
                #WordNet synsets for dependent
                dep_a, dep_b = wn.synsets(dep_a)[0], wn.synsets(dep_b)[0]

                #path based similarity (Li et. al) for dependent
                dep = dep_a.path_similarity(dep_b)

                dep = 0 if dep is None else dep

            except Exception:
                dep = 0     

        return head + dep

    def _similarity_tag(self, tag_a, tag_b):
        
        tag_a_id, tag_b_id = self._tags[tag_a], self._tags[tag_b] 
        score = self._tag_correlation_matrix[tag_a_id,tag_b_id]
        
        return score

    #TODO: Implement _similarity_graph method to find similarity between the tree structure of the documents
    # def _similarity_graph(pair_A, pair_B): 

    def semantic_similarity(self, document_1, document_2):

        #parsing documets using spaCy English language parser
        tokens_1,tokens_2 = self._parser(document_1), self._parser(document_2)

        #seperating dependency pairs and tags from tokens
        pairs_1 = [(token.head,token.dep_,token) for token in tokens_1]
        pairs_2 = [(token.head,token.dep_,token) for token in tokens_2]

        score = 0

        #calculating score 
        for pair_A in pairs_1:

            for pair_B in pairs_2:

                score += self._similarity_word(pair_A, pair_B) * self._similarity_tag(pair_A[1], pair_B[1])

        #averaging score 
        score = score / (len(tokens_1) + len(tokens_2))

        return score 