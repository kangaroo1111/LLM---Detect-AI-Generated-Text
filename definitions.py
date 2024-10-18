import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
from string import punctuation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LMHeadModel:

    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    
    def get_next_word_probabilities(self, sentence, top_k=500):

        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)
        
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()

        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))

class LLM_model:
    def __init__(self, name, size, with_prompt = False):
        self.name = name
        self.max_input_len = size
        self.with_prompt = with_prompt

def true_next_word(essay, n):
    word= ''
    i = n
    while str.isalpha(essay[i]) == False and str.isnumeric(essay[i]) == False:
        if essay[i] in punctuation:
            word+=essay[i]
            return word
        else:
            i+=1
    while essay[i] != ' ' and not(essay[i] in punctuation):
        word+=essay[i]
        i+=1
    return word


def return_index(element, list):
    i=0
    while i < len(list):
        if list[i] == element:
            return i
        else:
            i+=1
    if i == len(list):
        return -10000

def rths(r, list):
    rths = []
    for i in range(len(list)):
        if type(list[i]) is tuple:
            if len(list[i]) > r:
                rths.append(list[i][r])
            else:
                rths.append('')
        else:
            rths.append('')
    return rths


def predict_v2(text, prompt, ai):
    probability = []
    model = LMHeadModel(f'{ai.name}')
    for i in range(len(text)):
        if ai.with_prompt:
            feed = prompt + text[:i]
        else:
            feed = text[:i]
        p_words = model.get_next_word_probabilities(feed[max(0, len(feed)-ai.max_input_len -1):], top_k=500)
        index = return_index(true_next_word(text, i), rths(0, p_words))
        #print(f'index: {index}')
        if index >= 0:
            probability.append(p_words[index][1])
            #print(f'value: {p_words[index][1]}')
        else:
            probability.append(1e-10)
        #print(f'probability: {probability}')
    return probability


model1 = LLM_model('gpt2', 4096, with_prompt = True)