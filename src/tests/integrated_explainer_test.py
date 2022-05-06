import sys
sys.path.insert(0,"..")
 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
from methods.integrated_grads import IntegratedGradientsExplainer


if __name__ == '__main__':
    
    model = AutoModelForSequenceClassification.from_pretrained('../../models/checkpoint-5000')
    tokenizer = AutoTokenizer.from_pretrained('../../models/checkpoint-5000')
    explainer = IntegratedGradientsExplainer(model, tokenizer)
    
    data = pd.read_csv('../../data/test.csv')
    text = list(data['text'].values)

    print(explainer.multi_faithfulness(text[0:10]))
