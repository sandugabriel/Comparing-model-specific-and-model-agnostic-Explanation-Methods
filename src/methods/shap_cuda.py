import pandas as pd
import numpy as np
import shap
from tqdm import tqdm
import torch

class ShapExplainerCuda:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = shap.Explainer(self.model)
        self.device = device

    
    def _predict_proba(self, sentence):
        probas = []
        tokenized_sentence = torch.from_numpy(np.array(self.tokenizer(sentence, truncation = True, max_length = 512)['input_ids'])[1:-1]).float().to(self.device)
        print(tokenized_sentence.shape)
        for label in self.model(tokenized_sentence)[0]:
            probas.append(label['score'])
        
        return np.array(probas)
    
    def _explain(self, sentence):
        return self.explainer(sentence)
    
    def single_faithfulness(self, text):
        # calculate word attributions
        pred_class = np.argmax(self._predict_proba(text))
        word_attributions = self.explainer([text])
        coefs = []
        if word_attributions.shape[1] > 512:
            for x in word_attributions.values[0][:,pred_class][:512]:
                coefs.append(x)
        else:
            for x in word_attributions.values[0][:,pred_class]:
                coefs.append(x)
        # print(len(coefs))
        coefs = np.array(coefs[1:-1])
        tokens = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]
        base = np.zeros(tokens.shape[0])

        #find predicted class
        # pred_class = np.argmax(predict_proba(text))
        x = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]

        #find indexs of coefficients in decreasing order of value
        ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
        pred_probs = np.zeros(x.shape[0])

        for ind in np.nditer(ar):
            x_copy = x.copy()
            # print(x.shape)
            # print(base.shape)
            x_copy[ind] = base[ind]
            decoded_copy = self.tokenizer.decode(x_copy)
            x_copy_pr = self._predict_proba(decoded_copy)
            pred_probs[ind] = x_copy_pr[pred_class]

        return -np.corrcoef(coefs, pred_probs)[0,1]

    def multi_faithfulness(self, texts):
        faithfulness_array = []
        for t in tqdm(texts):
            faithfulness_array.append(self.single_faithfulness(t))

        m = np.array(faithfulness_array)

        return np.mean(m)
    
    def single_monotonicity(self, text):
        pred_class = np.argmax(self._predict_proba(text))
        # calculate word attributions
        word_attributions = self.explainer([text])
        coefs = []
        print(word_attributions)
        if word_attributions.shape[1] > 512:
            for x in word_attributions.values[0][:,pred_class][:512]:
                coefs.append(x)
        else:
            for x in word_attributions.values[0][:,pred_class]:
                coefs.append(x)
        coefs = np.array(coefs[1:-1])
        tokens = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]
        base = np.zeros(tokens.shape[0])

        #find predicted class
        
        x = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]
        x_copy = base.copy()

        #find indexs of coefficients in increasing order of value
        ar = np.argsort(coefs)
        pred_probs = np.zeros(x.shape[0])
        for ind in np.nditer(ar):
            x_copy[ind] = x[ind]
            decoded_copy = self.tokenizer.decode(x_copy.astype(int))
            x_copy_pr = self._predict_proba(decoded_copy)
            pred_probs[ind] = x_copy_pr[pred_class]
        
        return np.all(np.diff(pred_probs[ar]) >= 0)
    
    def multi_monotonicity(self, texts):
        monotonicity_array = []
        for t in tqdm(texts):
            monotonicity_array.append(self.single_monotonicity(t))

        m = np.array(monotonicity_array)

        return np.mean(m)