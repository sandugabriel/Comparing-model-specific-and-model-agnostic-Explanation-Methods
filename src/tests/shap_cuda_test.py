import sys


sys.path.insert(0,"..")
from methods.shap_cuda import ShapExplainerCuda
from methods.shap import ShapExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
import torch


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained('../../models/checkpoint-5000')
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('../../models/checkpoint-5000')

    # pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, truncation = True, max_length = 512)
    # pipe = pipe.to(device)
    explainer = ShapExplainerCuda(model, tokenizer, device)

    data = pd.read_csv('../../data/test.csv')
    text = list(data['text'].values)

    print(explainer.multi_faithfulness(text))
