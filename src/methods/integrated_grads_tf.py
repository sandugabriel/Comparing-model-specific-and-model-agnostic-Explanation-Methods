import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from transformers_interpret import SequenceClassificationExplainer
from tqdm import tqdm
import tensorflow as tf

class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = SequenceClassificationExplainer(self._predict_proba, self.tokenizer)

    def embedding_model(self, batch_ids):
        batch_embedding = self.model.distilbert.embeddings(batch_ids)
        return batch_embedding

    def prediction_model(self, batch_embedding):
        attention_mask = tf.ones(batch_embedding.shape[:2])
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        head_mask = [None] * self.model.distilbert.num_hidden_layers

        transformer_output = self.model.distilbert.transformer(batch_embedding, attention_mask, head_mask, training = False, output_attentions = True, output_hidden_states = True, return_dict = True)[0]
        pooled_output = transformer_output[:, 0]
        pooled_output = self.model.pre_classifier(pooled_output)
        logits = self.model.classifier(pooled_output)
        return logits
    
    def get_embeddings(self, sentence):
        tokenized_text = self.tokenizer.prepare_seq2seq_batch([sentence], return_tensors='pt', padding = 'max_length', truncation = True, max_length = 128)
        batch_embedding = self.embedding_model(tokenized_text['input_ids'])
        s = batch_embedding.shape[1]

        baseline_ids = np.zeros((1, s), dtype=np.int64)
        baseline_embedding = self.embedding_model(baseline_ids)

        return baseline_embedding, batch_embedding, tokenized_text
    
    def _predict_proba(self, sentence):
        tokenized_text = self.tokenizer.prepare_seq2seq_batch(sentence, return_tensors='pt')
        batch_embedding = self.embedding_model(tokenized_text['input_ids'])

        return self.prediction_model(batch_embedding)
    
    def single_faithfulness(self, text):
        # calculate word attributions
        pred_class = np.argmax(self._predict_proba(text))
        word_attributions = self.explainer(text)
        coefs = []
        # if word_attributions.shape[1] > 512:
        #     for x in word_attributions.values[0][:,pred_class][:512]:
        #         coefs.append(x)
        # else:
        #     for x in word_attributions.values[0][:,pred_class]:
        #         coefs.append(x)
        # print(len(coefs))
        for v in word_attributions:
            coefs.append(v[1])
        coefs = np.array(coefs[1:-1])
        tokens = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]
        base = np.zeros(tokens.shape[0])

        #find predicted class
        # pred_class = np.argmax(predict_proba(text))
        x = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]

        #find indexs of coefficients in decreasing order of value
        ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
        pred_probs = np.zeros(x.shape[0])
        # print(ar)
        for ind in np.nditer(ar):
            x_copy = x.copy()
            # print(x.shape)
            # print(base.shape)
            x_copy[ind] = base[ind]
            decoded_copy = self.tokenizer.decode(x_copy)
            x_copy_pr = self._predict_proba(decoded_copy)
            pred_probs[ind] = x_copy_pr[pred_class]
            
        return -np.corrcoef(coefs, pred_probs)[0,1]

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
    
    def multi_faithfulness(self, texts):
        faithfulness_array = []
        for t in tqdm(texts):
            faithfulness_array.append(self.single_faithfulness(t))

        m = np.array(faithfulness_array)

        return np.mean(m)
    
    def multi_monotonicity(self, texts):
        monotonicity_array = []
        for t in tqdm(texts):
            monotonicity_array.append(self.single_monotonicity(t))

        m = np.array(monotonicity_array)

        return np.mean(m)