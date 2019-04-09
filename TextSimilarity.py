#!/usr/bin/env python
# coding: utf-8

import spacy
from nltk.corpus import wordnet as wn

np = spacy.load('en')

def _similarity_word(pair_A, pair_B):
    
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
    
#TODO: Implent relation matrix between tags and generate score with respect to the matrix
_similarity_tag = lambda a, b : 1 if a == b else 0

#TODO: Implement _similarity_graph method to find similarity between the tree structure of the documents
# def _similarity_graph(pair_A, pair_B):

def semantic_similarity(document_1, document_2):
    
    #parsing documets using spaCy English language parser
    tokens_1,tokens_2 = np(document_1), np(document_2)
    
    #seperating dependency pairs and tags from tokens
    pairs_1 = [(token.head,token.dep_,token) for token in tokens_1]
    pairs_2 = [(token.head,token.dep_,token) for token in tokens_2]
    
    score = 0
    
    #calculating score 
    for pair_A in pairs_1:
        
        for pair_B in pairs_2:
            
            score += _similarity_word(pair_A, pair_B) * _similarity_tag(pair_A[1], pair_B[1])
    
    #averaging score 
    score = score / (len(tokens_1) + len(tokens_2))
    
    return score 
