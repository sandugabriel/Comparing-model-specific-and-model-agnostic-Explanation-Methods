import sys
sys.path.insert(0,"..")
 
from methods.lime import LimeExplainer
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig
import pandas as pd


if __name__ == '__main__':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = TFDistilBertForSequenceClassification.from_pretrained('../../models/tfdistilbert_dizertatie3', config=config)

    explainer = LimeExplainer(model, tokenizer)

    data = pd.read_csv('../../data/test.csv')
    text = list(data['text'].values)

    print(explainer.multi_faithfulness(text[0:10]))
