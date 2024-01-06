import pandas as pd


def load_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = [1 if label == "spam" else 0 for label in df['label'].tolist()]
    return texts, labels