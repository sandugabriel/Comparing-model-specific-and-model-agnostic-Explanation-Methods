import pandas as pd
import numpy as np
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
import tensorflow as tf

class LimeExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = LimeTextExplainer(class_names = ['Non-toxic', 'Toxic'])
    
    def _embedding_model(self, batch_ids):
        batch_embedding = self.model.distilbert.embeddings(batch_ids)
        return batch_embedding

    def _prediction_model(self, batch_embedding):
        attention_mask = tf.ones(batch_embedding.shape[:2])
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        head_mask = [None] * self.model.distilbert.num_hidden_layers

        transformer_output = self.model.distilbert.transformer(batch_embedding, attention_mask, head_mask, training=False, output_attentions = True, output_hidden_states = True, return_dict = True)[0]
        pooled_output = transformer_output[:, 0]
        pooled_output = self.model.pre_classifier(pooled_output)
        logits = self.model.classifier(pooled_output)
        prediction = tf.nn.softmax(logits)
        return prediction.numpy()
    
    def _predict_proba(self, sentences):
        tokenized_text = self.tokenizer.prepare_seq2seq_batch(sentences, return_tensors='pt')
        batch_embedding = self._embedding_model(tokenized_text['input_ids'])

        return self._prediction_model(batch_embedding)
    
    def single_faithfulness(self, text):
        # calculate word attributions
        pred_class = np.argmax(self._predict_proba(text))
        tokens = np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]
        t = []
        for x in np.array(self.tokenizer(text, truncation = True, max_length = 512)['input_ids'])[1:-1]:
            t.append(self.tokenizer.decode([x]))

        exp = self.explainer.explain_instance(" ".join(t), self._predict_proba, num_samples=500, labels = [0,1], num_features = 10000)
        coefs = []
        for word in t:
            word = word.strip("#")
            for elem in exp.domain_mapper.map_exp_ids(exp.local_exp[pred_class]):
                if elem[0] == word:
                    coefs.append(elem[1])

        if len(coefs) > 512:
            coefs = coefs[:512]
        # print(len(coefs))
        coefs = np.array(coefs)
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
            x_copy[ind] = base[ind]
            decoded_copy = self.tokenizer.decode(x_copy)
            x_copy_pr = self._predict_proba(decoded_copy)[0]
            pred_probs[ind] = x_copy_pr[pred_class]

        return -np.corrcoef(coefs, pred_probs)[0,1]
    
    def multi_faithfulness(self, texts):
        faithfulness_array = []
        for t in tqdm(texts):
            faithfulness_array.append(self.single_faithfulness(t))

        m = np.array(faithfulness_array)

        return np.mean(m)