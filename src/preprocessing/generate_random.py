import pandas as pd
import numpy as np

def generate_random_labels(data_path, label_col):
    data = pd.read_csv(data_path)

    data[label_col] = np.random.shuffle(data[label_col].values)
    return data

if __name__ == "__main__":
    randomized_train = generate_random_labels("../../data/train.csv")
    randomized_train.to_csv('randomized.csv')