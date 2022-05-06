import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from path_explain import PathExplainerTF
import random


class ExpectedGradientsExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = PathExplainerTF(self.prediction_model)
        
    
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
    
    def single_faithfulness(self, text, baseline_texts):
        tokenized_train = self.tokenizer.prepare_seq2seq_batch(random.sample(baseline_texts, 10000), return_tensors = 'pt', max_length = 128, padding = True)
        batch_embeddings = self.embedding_model(tokenized_train['input_ids'])
        pred_class = np.argmax(self._predict_proba(text))
        baseline_embedding, batch_embedding, tokenized_text = self.get_embeddings(text)
        print(batch_embeddings.shape)
        attributions = self.explainer.attributions(inputs=batch_embedding,
                                        baseline=batch_embeddings,
                                        batch_size=4,
                                        num_samples=batch_embeddings.shape[0],
                                        use_expectation=True,
                                        output_indices=1)

        attributions = np.mean(attributions, axis = 2)
        coefs = attributions[:tokenized_text['input_ids'].numpy()[0].shape[0]]
        if len(coefs) > 512:
            coefs = coefs[:512]
        
        x = tokenized_text['input_ids'].numpy()[0]
        coefs = np.array(coefs)
        base = np.zeros(x.shape[0])


        #find indexs of coefficients in decreasing order of value
        ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
        pred_probs = np.zeros(x.shape[0])
        print(coefs.shape)
        print(ar.shape)
        print(x.shape)

        for ind in np.nditer(ar):
            x_copy = x.copy()
            x_copy[ind] = base[ind]
            decoded_copy = self.tokenizer.decode(x_copy)
            x_copy_pr = self._predict_proba(decoded_copy)[0]
            pred_probs[ind] = x_copy_pr[pred_class]

        return -np.corrcoef(coefs, pred_probs)[0,1]

    def multi_faithfulness(self, texts, baseline_texts):
        faithfulness_array = []
        for t in tqdm(texts):
            faithfulness_array.append(self.single_faithfulness(t, baseline_texts))

        m = np.array(faithfulness_array)

        return np.mean(m)

    def single_monotonicity(self, text, baseline_texts):
        tokenized_train = self.tokenizer.prepare_seq2seq_batch(random.sample(baseline_texts, 50), return_tensors = 'pt', max_length = 128, padding = True)
        batch_embeddings = self.embedding_model(tokenized_train['input_ids'])
        pred_class = np.argmax(self._predict_proba(text))
        baseline_embedding, batch_embedding, tokenized_text = self.get_embeddings(text)

        attributions = self.explainer.attributions(inputs=batch_embedding,
                                        baseline=batch_embeddings,
                                        batch_size=4,
                                        num_samples=128,
                                        use_expectation=True,
                                        output_indices=1)

        attributions = np.mean(attributions, axis = 2)
        coefs = attributions[:tokenized_text['input_ids'].numpy()[0].shape[0]]
        if len(coefs) > 512:
            coefs = coefs[:512]
        
        x = tokenized_text['input_ids'].numpy()[0]
        coefs = np.array(coefs)
        base = np.zeros(x.shape[0])

        x_copy = base.copy()

        #find indexs of coefficients in increasing order of value
        ar = np.argsort(coefs)
        pred_probs = np.zeros(x.shape[0])
        for ind in np.nditer(ar):
            x_copy[ind] = x[ind]
            decoded_copy = self.tokenizer.decode(x_copy.astype(int))
            x_copy_pr = self._predict_proba(decoded_copy).numpy()[0]
            pred_probs[ind] = x_copy_pr[pred_class]
        
        return np.all(np.diff(pred_probs[ar]) >= 0)
    
    def multi_monotonicity(self, texts, baseline_texts):
        monotonicity_array = []
        for t in tqdm(texts):
            monotonicity_array.append(self.single_monotonicity(t, baseline_texts))

        m = np.array(monotonicity_array)

        return np.mean(m)