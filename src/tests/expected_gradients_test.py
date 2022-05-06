import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_GPU_ALLOCATOR']= 'cuda_malloc_async'
sys.path.insert(0,"..")

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification
import pandas as pd
from methods.expected_grads import ExpectedGradientsExplainer


if __name__ == '__main__':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = TFDistilBertForSequenceClassification.from_pretrained('../../models/tfdistilbert_dizertatie3', config=config)
    explainer = ExpectedGradientsExplainer(model, tokenizer)

    data = pd.read_csv('../../data/test.csv')
    train = pd.read_csv('../../data/train.csv')
    text = list(data['text'].values)
    train_lst = list(train['text'].values)

    print(explainer.multi_faithfulness(text[0:10], train_lst))
