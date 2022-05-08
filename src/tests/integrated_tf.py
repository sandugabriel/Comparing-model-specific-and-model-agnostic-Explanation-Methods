import sys
sys.path.insert(0,"..")
 
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification
import pandas as pd
from methods.integrated_grads_tf import IntegratedGradientsExplainer


if __name__ == '__main__':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = TFDistilBertForSequenceClassification.from_pretrained('../../models/tfdistilbert_dizertatie3', config=config)
    explainer = IntegratedGradientsExplainer(model, tokenizer)
    # print('test')
    data = pd.read_csv('../../data/test.csv')
    text = list(data['text'].values)

    print(explainer.multi_faithfulness(text[0:10]))
